"""
ethsim_generic.py - public generic ETH T-DCA engine (Ranveer Verma, github.com/ranveer9)
========================================================================================
Same event model and code path as the privately delivered engine used for the Blitz ETH
verification, with EVERY strategy, cost and exchange-filter setting loaded at run time:

    import ethsim_generic as X
    X.configure("presets.json")   # private file, not published - call BEFORE any simulation

The exact configuration (ladder distances, allocations, wallet usage, leverage, take-profit,
reset threshold, fees, slippage, maintenance, tick and lot filters) is private to Blitz Trading
and not included. Without it the published results cannot be reproduced.

Model summary: minute loop (risk update at each 1-minute mark close, funding before the minute's
trades, then trades paired with that close); segment-crossing ladder and take-profit fills with
marks interpolated by price fraction; flat maintenance proxy; exit-net terminal equity.
Modes: 0 reference; 1 one-tick trade-through; 2 volume at/through >= k x size; 3 latency.
Modes 1-3 are sensitivity analyses (changed assumptions), not the reference model.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# float state
(W, BASE, P1, QTY, COST, PEAK, MDD, MINROOM, FEES, FUND, SLIP, TURN, TPSHORT, WORST, BLKMIN,
 PREVP, PREVM, LASTMARK, VTP, VTPT, YPEAK, YDD, YMINROOM, YMAXHOLD, YLASTE, YLASTW, YLASTMARK,
 MAXHOLD) = range(28)
NF = 28
# int state
(OPEN, NEXT, DEPTH, TERM, INVALID, LIQ, TPC, RESETS, MAXSTEP, OPENTS, FIDX, NCYC, LASTTS, FIRSTTS,
 TRADES, HAVEPREV, NEQ, LASTDAY, YMAXSTEP, RUNGACT, TPACT, REENTRY, FLATFUND, INVALIDTS, YOBS, RISKTS,
 OVERFLOW) = range(27)
NI = 27
# cycle records
CF_WORST, CF_M, CF_C, CF_COST, CF_QTY = range(5)
NCF = 5
CI_OPEN, CI_CLOSE, CI_DEPTH, CI_LIQ = range(4)
NCI = 4

# ---- configuration: populated only by configure(); numba reads these at first compilation ----
TICK = float("nan")
MMR = float("nan")
TPP = float("nan")          # take-profit price return
ISLIP = float("nan")        # initial entry slippage, bps
LSLIP = float("nan")        # later entry slippage, bps
QSTEP = float("nan")        # quantity step
QMIN = float("nan")         # minimum quantity
NMIN = float("nan")         # minimum executed notional
_CONFIGURED = False


def configure(path) -> dict:
    """Load the private presets.json. Must be called once, before any simulation is compiled."""
    import json as _json
    global TICK, MMR, TPP, ISLIP, LSLIP, QSTEP, QMIN, NMIN, _CONFIGURED
    cfg = _json.loads(open(path, encoding="utf-8").read())
    c = cfg["configuration"]
    vals = (float(c["price_tick"]), float(c["maintenance_margin_rate"]), float(c["tp_price_return"]),
            float(c["initial_slippage_bps"]), float(c["later_slippage_bps"]), float(c["quantity_step"]),
            float(c["min_quantity"]), float(c["min_notional"]))
    if _CONFIGURED and vals != (TICK, MMR, TPP, ISLIP, LSLIP, QSTEP, QMIN, NMIN):
        raise RuntimeError("ethsim_generic: already configured and compiled with different values")
    TICK, MMR, TPP, ISLIP, LSLIP, QSTEP, QMIN, NMIN = vals
    _CONFIGURED = True
    return cfg


def require_configured() -> None:
    if not _CONFIGURED:
        raise RuntimeError("ethsim_generic: call configure(<private presets.json>) first. "
                           "The configuration is private and not included in this repository.")


@njit(cache=False)
def pdown(x):
    return np.floor(x / TICK + 1e-12) * TICK


@njit(cache=False)
def pup(x):
    return np.ceil(x / TICK - 1e-12) * TICK


@njit(cache=False)
def qfloor(notional, price):
    return np.floor(notional / price / QSTEP + 1e-12) * QSTEP


@njit(cache=False)
def _avg(fs, s):
    return fs[s, COST] / fs[s, QTY]


@njit(cache=False)
def _liq_line(fs, s):
    q = fs[s, QTY]
    num = q * _avg(fs, s) - fs[s, W]
    den = q * (1.0 - MMR)
    if num <= 0.0 or den <= 0.0:
        return 0.0
    return num / den


@njit(cache=False)
def _exit_net(fs, ist, s, mark, slip_bps, tp_fee):
    if ist[s, OPEN] == 0:
        return fs[s, W]
    ex = pdown(mark * (1.0 - slip_bps / 10000.0))
    q = fs[s, QTY]
    return fs[s, W] + q * (ex - _avg(fs, s)) - q * ex * tp_fee


@njit(cache=False)
def update_risk(fs, ist, s, mark, ts, eq_ts, eq_v):
    """Returns True if liquidated."""
    opn = ist[s, OPEN] == 1
    if opn:
        e = fs[s, W] + fs[s, QTY] * (mark - _avg(fs, s))
    else:
        e = fs[s, W]
    if e > fs[s, PEAK]:
        fs[s, PEAK] = e
    dd = 1.0 - e / fs[s, PEAK]
    if dd > fs[s, MDD]:
        fs[s, MDD] = dd
    # annual observer
    if e > fs[s, YPEAK]:
        fs[s, YPEAK] = e
    ydd = 1.0 - e / fs[s, YPEAK]
    if ydd > fs[s, YDD]:
        fs[s, YDD] = ydd
    fs[s, YLASTE] = e
    fs[s, YLASTW] = fs[s, W]
    fs[s, YLASTMARK] = mark
    fs[s, LASTMARK] = mark
    ist[s, RISKTS] = ts                     # state.last_ts: timestamp of the last risk update
    if ts > ist[s, LASTTS]:
        ist[s, LASTTS] = ts
    day = ts // 86400000
    if day != ist[s, LASTDAY]:
        n = ist[s, NEQ]
        if n < eq_ts.shape[1]:
            eq_ts[s, n] = ts
            eq_v[s, n] = e
            ist[s, NEQ] = n + 1
        ist[s, LASTDAY] = day
    if opn:
        room = (mark - _liq_line(fs, s)) / mark
        if room < fs[s, MINROOM]:
            fs[s, MINROOM] = room
        if room < fs[s, YMINROOM]:
            fs[s, YMINROOM] = room
        ist[s, YOBS] = 1
        h = float(ts - ist[s, OPENTS])
        if h > fs[s, MAXHOLD]:
            fs[s, MAXHOLD] = h
        if h > fs[s, YMAXHOLD]:
            fs[s, YMAXHOLD] = h
        if ist[s, DEPTH] > ist[s, YMAXSTEP]:
            ist[s, YMAXSTEP] = ist[s, DEPTH]
        if e <= fs[s, QTY] * mark * MMR + 1e-10:
            return True
    return False


@njit(cache=False)
def _record(fs, ist, s, ts, liq, cf, ci):
    c = ist[s, NCYC]
    if c < cf.shape[1]:
        cf[s, c, CF_WORST] = fs[s, WORST]
        cf[s, c, CF_M] = fs[s, COST] / 5.0
        cf[s, c, CF_C] = fs[s, W]
        cf[s, c, CF_COST] = fs[s, COST]
        cf[s, c, CF_QTY] = fs[s, QTY]
        ci[s, c, CI_OPEN] = ist[s, OPENTS]
        ci[s, c, CI_CLOSE] = ts
        ci[s, c, CI_DEPTH] = ist[s, DEPTH]
        ci[s, c, CI_LIQ] = liq
        ist[s, NCYC] = c + 1
    else:
        ist[s, OVERFLOW] = 1


@njit(cache=False)
def _liquidate(fs, ist, s, ts, cf, ci):
    _record(fs, ist, s, ts, 1, cf, ci)
    fs[s, W] = 0.0
    ist[s, LIQ] += 1
    ist[s, OPEN] = 0
    fs[s, QTY] = 0.0
    fs[s, COST] = 0.0
    ist[s, TERM] = 1


@njit(cache=False)
def _try_open(fs, ist, s, price, mark, ts, cov, alloc, lvl_p, lvl_q, lvl_f, lvl_v,
              U, LEV, FEE0, FEE1, RATIO, eq_ts, eq_v, cf, ci):
    """Full-ladder preview + initial entry. Returns False if execution invalid (run stops)."""
    R = cov.shape[0]
    base = fs[s, BASE]
    p1 = pup(price * (1.0 + ISLIP / 10000.0))
    q0 = qfloor(base * U * alloc[0] * LEV, price)
    need = 0.0
    ok = q0 >= QMIN and q0 * p1 >= NMIN
    n0 = q0 * p1
    need += n0 / LEV + n0 * FEE0
    for k in range(1, R):
        L = pdown(p1 * (1.0 - cov[k]))
        q = qfloor(base * U * alloc[k] * LEV, L)
        f = pup(L * (1.0 + LSLIP / 10000.0))
        lvl_p[s, k] = L
        lvl_q[s, k] = q
        lvl_f[s, k] = f
        lvl_v[s, k] = 0.0
        if q < QMIN or q * f < NMIN:
            ok = False
        need += q * f / LEV + q * f * FEE1
    if lvl_p[s, R - 1] < RATIO * p1:
        ok = False
    tol = max(1e-9, fs[s, W] * 1e-12)
    if fs[s, W] + tol < need:
        ok = False
    if not ok:
        ist[s, INVALID] = 1
        ist[s, INVALIDTS] = ts
        ist[s, TERM] = 1
        return False
    fee = n0 * FEE0
    fs[s, W] -= fee
    fs[s, FEES] += fee
    fs[s, TURN] += n0
    sl = p1 - price
    if sl > 0.0:
        fs[s, SLIP] += sl * q0
    fs[s, P1] = p1
    fs[s, QTY] = q0
    fs[s, COST] = q0 * p1
    fs[s, WORST] = 0.0
    fs[s, BLKMIN] = np.inf
    fs[s, VTP] = 0.0
    fs[s, VTPT] = -1.0
    ist[s, OPEN] = 1
    ist[s, NEXT] = 1
    ist[s, DEPTH] = 1
    if ist[s, MAXSTEP] < 1:
        ist[s, MAXSTEP] = 1
    ist[s, OPENTS] = ts
    return True


@njit(cache=False)
def _fill_level(fs, ist, s, k, lvl_p, lvl_q, lvl_f, FEE1):
    q = lvl_q[s, k]
    f = lvl_f[s, k]
    n = q * f
    fs[s, QTY] += q
    fs[s, COST] += q * f
    fee = n * FEE1
    fs[s, W] -= fee
    fs[s, FEES] += fee
    fs[s, TURN] += n
    sl = f - lvl_p[s, k]
    if sl > 0.0:
        fs[s, SLIP] += sl * q
    ist[s, NEXT] = k + 1
    ist[s, DEPTH] = k + 1
    if k + 1 > ist[s, MAXSTEP]:
        ist[s, MAXSTEP] = k + 1


@njit(cache=False)
def _close_tp(fs, ist, s, T, ts, slip_bps, TPFEE, RESET, eq_ts, eq_v, cf, ci, mark):
    q = fs[s, QTY]
    avg = _avg(fs, s)
    fill = pdown(T * (1.0 - slip_bps / 10000.0))
    exn = q * fill
    fee = exn * TPFEE
    fs[s, W] += q * (fill - avg) - fee
    fs[s, FEES] += fee
    fs[s, TURN] += exn
    sl = T - fill
    if sl > 0.0:
        fs[s, SLIP] += sl * q
    sh = avg * (1.0 + TPP) - T
    if sh > 0.0:
        fs[s, TPSHORT] += sh * q
    ist[s, TPC] += 1
    _record(fs, ist, s, ts, 0, cf, ci)
    ist[s, OPEN] = 0
    fs[s, QTY] = 0.0
    fs[s, COST] = 0.0
    update_risk(fs, ist, s, mark, ts, eq_ts, eq_v)       # close_tp observation, now flat
    if fs[s, W] >= fs[s, BASE] * (1.0 + RESET) - 1e-10:
        fs[s, BASE] = fs[s, W]
        ist[s, RESETS] += 1


@njit(cache=False)
def _apply_funding(fs, ist, s, fmark, fts, rate, pos_mult, neg_mult, eq_ts, eq_v, cf, ci):
    if ist[s, OPEN] == 0:
        ist[s, FLATFUND] += 1
        return
    if update_risk(fs, ist, s, fmark, fts, eq_ts, eq_v):
        _liquidate(fs, ist, s, fts, cf, ci)
        return
    mult = pos_mult if rate >= 0.0 else neg_mult
    pay = fs[s, QTY] * fmark * rate * mult
    fs[s, W] -= pay
    fs[s, FUND] += pay
    if update_risk(fs, ist, s, fmark, fts, eq_ts, eq_v):
        _liquidate(fs, ist, s, fts, cf, ci)


@njit(cache=False)
def _sample_mae(fs, s, p):
    if p < np.inf:
        v = fs[s, QTY] * (p - _avg(fs, s))
        if v < fs[s, WORST]:
            fs[s, WORST] = v


@njit(cache=False)
def _path_hits(fs, s, ma, mb):
    lq = _liq_line(fs, s)
    return lq > 0.0 and min(ma, mb) <= lq, lq


@njit(cache=False)
def process_day(prices, stamps, qtys, m0_ts, m_lo, m_hi, closes, fts, frates, fmarks,
                cov, alloc, U, LEV, FEE0, FEE1, TPFEE, RESET, RATIO,
                p_mode, p_pos, p_neg, p_slip, p_k, p_lat,
                fs, ist, lvl_p, lvl_q, lvl_f, lvl_v, cf, ci, eq_ts, eq_v):
    R = cov.shape[0]
    n = prices.shape[0]
    nf = fts.shape[0]
    for s in range(fs.shape[0]):
        mode = p_mode[s]
        slip = p_slip[s]
        t = 0
        for mi in range(m_lo, m_hi):
            if ist[s, TERM] == 1:
                break
            mts = m0_ts + mi * 60000
            mark = closes[mi]
            # 1. minute-start risk observation
            if update_risk(fs, ist, s, mark, mts, eq_ts, eq_v):
                _liquidate(fs, ist, s, mts, cf, ci)
                break
            # 2. funding before the minute's trades
            while ist[s, FIDX] < nf and fts[ist[s, FIDX]] < mts + 60000:
                j = ist[s, FIDX]
                _apply_funding(fs, ist, s, fmarks[j], fts[j], frates[j], p_pos[s], p_neg[s],
                               eq_ts, eq_v, cf, ci)
                ist[s, FIDX] = j + 1
                if ist[s, TERM] == 1:
                    break
            if ist[s, TERM] == 1:
                break
            # 3. trades of this minute
            fs[s, BLKMIN] = np.inf
            while t < n and (stamps[t] - m0_ts) // 60000 < mi:
                t += 1
            while t < n and (stamps[t] - m0_ts) // 60000 == mi:
                p = prices[t]
                ts = stamps[t]
                ist[s, TRADES] += 1
                if ist[s, FIRSTTS] == 0:
                    ist[s, FIRSTTS] = ts
                if ts > ist[s, LASTTS]:
                    ist[s, LASTTS] = ts
                if ist[s, OPEN] == 0:
                    if mode == 3 and ts < ist[s, REENTRY]:
                        fs[s, PREVP] = p
                        fs[s, PREVM] = mark
                        t += 1
                        continue
                    if not _try_open(fs, ist, s, p, mark, ts, cov, alloc, lvl_p, lvl_q, lvl_f, lvl_v,
                                     U, LEV, FEE0, FEE1, RATIO, eq_ts, eq_v, cf, ci):
                        break
                    if mode == 3:
                        ist[s, RUNGACT] = ts + p_lat[s]
                        ist[s, TPACT] = ts + p_lat[s]
                    if update_risk(fs, ist, s, mark, ts, eq_ts, eq_v):
                        _liquidate(fs, ist, s, ts, cf, ci)
                        break
                    fs[s, PREVP] = p
                    fs[s, PREVM] = mark
                    fs[s, BLKMIN] = np.inf
                    t += 1
                    continue
                P0 = fs[s, PREVP]
                M0 = fs[s, PREVM]
                P1v = p
                M1 = mark
                T = pdown(_avg(fs, s) * (1.0 + TPP))
                if p < fs[s, BLKMIN]:
                    fs[s, BLKMIN] = p
                if mode == 0 or mode == 1 or mode == 3:
                    tick = TICK if mode == 1 else 0.0
                    act_r = mode != 3 or ts >= ist[s, RUNGACT]
                    act_t = mode != 3 or ts >= ist[s, TPACT]
                    k = ist[s, NEXT]
                    desc = False
                    if act_r and k < R:
                        L = lvl_p[s, k]
                        if mode == 3:
                            desc = L >= P1v + tick - 1e-12          # active order: at or through
                        else:
                            desc = P1v < P0 and L >= P1v + tick - 1e-12 and L < P0 + 1e-12
                    asc = False
                    if act_t:
                        if mode == 3:
                            asc = P1v >= T + tick - 1e-12
                        else:
                            asc = P1v > P0 and P0 < T + tick and T + tick <= P1v + 1e-12
                    if desc:
                        _sample_mae(fs, s, fs[s, BLKMIN])
                        fs[s, BLKMIN] = np.inf
                        ma = M0
                        stop = False
                        while ist[s, NEXT] < R:
                            k = ist[s, NEXT]
                            L = lvl_p[s, k]
                            if mode == 3:
                                if L < P1v + tick - 1e-12:
                                    break
                            elif not (L >= P1v + tick - 1e-12 and L < P0 + 1e-12):
                                break
                            if P1v != P0:
                                fr = (L - P0) / (P1v - P0)
                                fr = min(1.0, max(0.0, fr))
                            else:
                                fr = 1.0
                            mL = M0 + (M1 - M0) * fr
                            hit, lq = _path_hits(fs, s, ma, mL)
                            if hit:
                                if update_risk(fs, ist, s, lq, ts, eq_ts, eq_v):
                                    _liquidate(fs, ist, s, ts, cf, ci)
                                    stop = True
                                    break
                            _fill_level(fs, ist, s, k, lvl_p, lvl_q, lvl_f, FEE1)
                            if update_risk(fs, ist, s, mL, ts, eq_ts, eq_v):
                                _liquidate(fs, ist, s, ts, cf, ci)
                                stop = True
                                break
                            ma = mL
                        if stop:
                            break
                        if mode == 3:
                            ist[s, TPACT] = ts + p_lat[s]
                        if update_risk(fs, ist, s, M1, ts, eq_ts, eq_v):
                            _liquidate(fs, ist, s, ts, cf, ci)
                            break
                    elif asc:
                        _sample_mae(fs, s, fs[s, BLKMIN])
                        fs[s, BLKMIN] = np.inf
                        TT = T + tick
                        if P1v != P0 and mode != 3:
                            fr = min(1.0, max(0.0, (TT - P0) / (P1v - P0)))
                        else:
                            fr = 1.0
                        mT = M0 + (M1 - M0) * fr
                        hit, lq = _path_hits(fs, s, M0, mT)
                        if hit:
                            if update_risk(fs, ist, s, lq, ts, eq_ts, eq_v):
                                _liquidate(fs, ist, s, ts, cf, ci)
                                break
                        if update_risk(fs, ist, s, mT, ts, eq_ts, eq_v):
                            _liquidate(fs, ist, s, ts, cf, ci)
                            break
                        _close_tp(fs, ist, s, T, ts, slip, TPFEE, RESET, eq_ts, eq_v, cf, ci, mT)
                        if mode == 3:
                            ist[s, REENTRY] = ts + p_lat[s]
                else:
                    # mode 2: volume-aware, sequential ladder, fills at trade j with its mark
                    k = ist[s, NEXT]
                    filled = False
                    while k < R:
                        L = lvl_p[s, k]
                        if p <= L + 1e-12:
                            lvl_v[s, k] += qtys[t]
                        if p <= L + 1e-12 and lvl_v[s, k] >= p_k[s] * lvl_q[s, k]:
                            if not filled:
                                _sample_mae(fs, s, fs[s, BLKMIN])
                                fs[s, BLKMIN] = np.inf
                            _fill_level(fs, ist, s, k, lvl_p, lvl_q, lvl_f, FEE1)
                            filled = True
                            if update_risk(fs, ist, s, M1, ts, eq_ts, eq_v):
                                _liquidate(fs, ist, s, ts, cf, ci)
                                break
                            k += 1
                        else:
                            break
                    if ist[s, TERM] == 1:
                        break
                    if filled:
                        fs[s, VTP] = 0.0
                    else:
                        T = pdown(_avg(fs, s) * (1.0 + TPP))
                        if fs[s, VTPT] != T:
                            fs[s, VTPT] = T
                            fs[s, VTP] = 0.0
                        if p >= T - 1e-12:
                            fs[s, VTP] += qtys[t]
                            if fs[s, VTP] >= p_k[s] * fs[s, QTY]:
                                _sample_mae(fs, s, fs[s, BLKMIN])
                                fs[s, BLKMIN] = np.inf
                                if update_risk(fs, ist, s, M1, ts, eq_ts, eq_v):
                                    _liquidate(fs, ist, s, ts, cf, ci)
                                    break
                                _close_tp(fs, ist, s, T, ts, slip, TPFEE, RESET, eq_ts, eq_v, cf, ci, M1)
                fs[s, PREVP] = p
                fs[s, PREVM] = mark
                t += 1
            if ist[s, TERM] == 1:
                break
            # unsampled block at minute end, valued with the current position
            if ist[s, OPEN] == 1:
                _sample_mae(fs, s, fs[s, BLKMIN])
            fs[s, BLKMIN] = np.inf
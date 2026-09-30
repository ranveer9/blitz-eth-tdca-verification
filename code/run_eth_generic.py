"""
run_eth_generic.py - PUBLIC runner (needs the private presets.json; not reproducible without it).
One pass over the ETHUSDT aggTrades, 8 simulations (compiled engine src/ethsim.py)

  baseline, funding, slip1, slip2      reference model (Phase A), per presets.json scenarios
  F1_tick, F2_vol1x, F3_vol2x, L1_lat1s  Phase B sensitivities on baseline costs

Usage:  python run_eth.py            (full run)
        python run_eth.py --days 30  (quick smoke test on the first 30 days; not comparable)
Output: output/private/  results_eth.json, comparison.json, cycles_<sim>.csv, equity_<sim>.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))
sys.path.insert(0, str(BASE_DIR / "src"))
import ethsim_generic as X  # noqa: E402

HO = BASE_DIR / "data" / "handover"
EXTRACTED = BASE_DIR / "data" / "extracted"
OUT = BASE_DIR / "output" / "private"
START_MS, END_MS = 1609459200000, 1787616000000
MAXC, MAXEQ = 80000, 2300


def verify_handover() -> None:
    bad = []
    for line in open(HO / "SHA256SUMS.txt", encoding="utf-8"):
        h, f = line.split(None, 1)
        p = HO / f.strip()
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            bad.append(f.strip())
    if bad:
        print("HANDOVER HASH CHECK FAILED:", bad)
        sys.exit(2)
    print("Handover files verified against SHA256SUMS (17/17)")


def load_day(path: Path):
    df = pd.read_csv(path, usecols=[1, 2, 5], header=0, names=["p", "q", "t"],
                     dtype={"p": np.float64, "q": np.float64, "t": np.int64}, engine="c")
    p = df["p"].to_numpy(np.float64)
    q = df["q"].to_numpy(np.float64)
    t = df["t"].to_numpy(np.int64)
    keep = (p > 0.0) & (t >= START_MS) & (t < END_MS)
    p, q, t = p[keep], q[keep], t[keep]
    o = np.argsort(t, kind="mergesort")          # stable; a no-op when already ordered
    return p[o], t[o], q[o]


def pct_stats(v: list[float], keys: tuple[str, ...]) -> dict:
    if not v:
        return {k: None for k in keys}
    a = np.array(v, dtype=np.float64)
    m = {"mean": float(np.mean(a)), "median": float(np.percentile(a, 50)), "p5": float(np.percentile(a, 5)),
         "p1": float(np.percentile(a, 1)), "worst": float(np.min(a)), "p95": float(np.percentile(a, 95)),
         "max": float(np.max(a))}
    return {k: m[k] for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    verify_handover()
    cfg = X.configure(HO / "presets.json")        # private configuration, loaded at run time
    X.require_configured()
    cand, c = cfg["candidate"], cfg["configuration"]
    cov = np.array(cand["entry_cum_cov"], np.float64)
    alloc = np.array(cand["allocation"], np.float64)
    scen = {x["id"]: x for x in cfg["scenarios"]}
    SIMS = [(sid, 0, scen[sid]["funding_positive_multiplier"], scen[sid]["funding_negative_multiplier"],
             scen[sid]["tp_slippage_bps"], 0.0, 0) for sid in ("baseline", "funding", "slip1", "slip2")]
    SIMS += [("F1_tick", 1, 0.0, 0.0, 0.0, 0.0, 0), ("F2_vol1x", 2, 0.0, 0.0, 0.0, 1.0, 0),
             ("F3_vol2x", 2, 0.0, 0.0, 0.0, 2.0, 0), ("L1_lat1s", 3, 0.0, 0.0, 0.0, 0.0, 1000)]
    S = len(SIMS)

    mk = pd.read_csv(HO / "market-data" / "ETHUSDT-mark-close-1m.csv.gz", dtype={"timestamp_ms": np.int64})
    mts = mk["timestamp_ms"].to_numpy(np.int64)
    closes = mk["mark_close"].to_numpy(np.float64)
    m0 = int(mts[0])
    assert np.all(np.diff(mts) == 60000), "mark minutes not consecutive"
    fj = json.loads((HO / "market-data" / "funding.json").read_text(encoding="utf-8"))
    fl = fj if isinstance(fj, list) else next(v for v in fj.values() if isinstance(v, list))
    fl = sorted(fl, key=lambda r: int(r["fundingTime"]))
    fts = np.array([int(r["fundingTime"]) for r in fl], np.int64)
    frates = np.array([float(r["fundingRate"]) for r in fl], np.float64)
    fmarks = np.array([float(r["markPrice"]) for r in fl], np.float64)
    print(f"Marks {len(mts):,} minutes; funding {len(fts):,} records")

    fs = np.zeros((S, X.NF)); ist = np.zeros((S, X.NI), np.int64)
    R = len(cov)
    lvl = [np.zeros((S, R)) for _ in range(4)]
    cf = np.zeros((S, MAXC, X.NCF)); ci = np.zeros((S, MAXC, X.NCI), np.int64)
    eq_ts = np.zeros((S, MAXEQ), np.int64); eq_v = np.zeros((S, MAXEQ))
    for s in range(S):
        fs[s, X.W] = fs[s, X.BASE] = fs[s, X.PEAK] = fs[s, X.YPEAK] = c["initial_wallet"]
        fs[s, X.MINROOM] = fs[s, X.YMINROOM] = np.inf
        ist[s, X.LASTDAY] = -1
    p_mode = np.array([x[1] for x in SIMS], np.int64); p_pos = np.array([x[2] for x in SIMS], np.float64)
    p_neg = np.array([x[3] for x in SIMS], np.float64); p_slip = np.array([x[4] for x in SIMS], np.float64)
    p_k = np.array([x[5] for x in SIMS], np.float64); p_lat = np.array([x[6] for x in SIMS], np.int64)

    dates = sorted(p.stem for p in EXTRACTED.glob("*.csv"))
    if args.days:
        dates = dates[:args.days]
    elif len(dates) != 2062:
        print(f"Expected 2062 daily files, found {len(dates)}"); sys.exit(3)

    years: dict[str, list[dict]] = {x[0]: [] for x in SIMS}
    cur_year = None
    snap: list[dict] = [{} for _ in range(S)]

    def exit_net_at(s: int, mark: float) -> float:
        return float(X._exit_net(fs, ist, s, mark, p_slip[s], c["tp_limit_fee"]))

    def marked_at(s: int, mark: float) -> float:
        if ist[s, X.OPEN] == 0:
            return float(fs[s, X.W])
        return float(fs[s, X.W] + fs[s, X.QTY] * (mark - fs[s, X.COST] / fs[s, X.QTY]))

    def open_obs(s: int, start_mark: float, start_ts: int, year: str) -> None:
        e = marked_at(s, start_mark)
        snap[s] = {"year": year, "start_marked": e, "start_exit": exit_net_at(s, start_mark),
                   "start_wallet": float(fs[s, X.W]), "fees": fs[s, X.FEES], "fund": fs[s, X.FUND],
                   "turn": fs[s, X.TURN], "slip": fs[s, X.SLIP], "tpc": int(ist[s, X.TPC])}
        fs[s, X.YPEAK] = e; fs[s, X.YDD] = 0.0; fs[s, X.YMINROOM] = np.inf; fs[s, X.YMAXHOLD] = 0.0
        ist[s, X.YMAXSTEP] = 0; ist[s, X.YOBS] = 0
        fs[s, X.YLASTE] = e; fs[s, X.YLASTW] = fs[s, X.W]; fs[s, X.YLASTMARK] = start_mark
        if ist[s, X.OPEN] == 1:                       # immediate observation of a carried position
            lq = float(X._liq_line(fs, s))
            fs[s, X.YMINROOM] = (start_mark - lq) / start_mark; ist[s, X.YOBS] = 1
            fs[s, X.YMAXHOLD] = float(start_ts - ist[s, X.OPENTS]); ist[s, X.YMAXSTEP] = ist[s, X.DEPTH]

    def close_obs(s: int) -> None:
        sp = snap[s]
        end_e = float(fs[s, X.YLASTE]); end_x = exit_net_at(s, float(fs[s, X.YLASTMARK]))
        years[SIMS[s][0]].append({
            "year": sp["year"], "start_wallet": sp["start_wallet"], "end_wallet": float(fs[s, X.YLASTW]),
            "profit_usd": float(fs[s, X.YLASTW]) - sp["start_wallet"],
            "marked_return_pct": (100.0 * (end_e / sp["start_marked"] - 1.0)) if sp["start_marked"] > 0 else -100.0,
            "exit_net_return_pct": (100.0 * (end_x / sp["start_exit"] - 1.0)) if sp["start_exit"] > 0 else -100.0,
            "max_drawdown_pct": 100.0 * float(fs[s, X.YDD]),
            "min_liquidation_room_pct": (100.0 * float(fs[s, X.YMINROOM])) if ist[s, X.YOBS] else None,
            "max_hold_days": float(fs[s, X.YMAXHOLD]) / 86400000.0, "max_step": int(ist[s, X.YMAXSTEP]),
            "fees_total": float(fs[s, X.FEES]) - sp["fees"], "funding_paid_signed": float(fs[s, X.FUND]) - sp["fund"],
            "turnover_notional": float(fs[s, X.TURN]) - sp["turn"], "slippage_cost": float(fs[s, X.SLIP]) - sp["slip"],
            "cycles_completed_tp": int(ist[s, X.TPC]) - sp["tpc"]})

    t0 = time.perf_counter(); tl = t0; tio = 0.0; final_mark = 0.0
    for di, d in enumerate(dates):
        a = time.perf_counter(); px, ts, qt = load_day(EXTRACTED / f"{d}.csv"); tio += time.perf_counter() - a
        if len(px) == 0:
            continue
        y = d[:4]
        if cur_year is None:
            for s in range(S):
                open_obs(s, float(px[0]), int(ts[0]), y)
            cur_year = y
        elif y != cur_year:
            for s in range(S):
                close_obs(s)
                open_obs(s, float(fs[s, X.LASTMARK]), int(ist[s, X.RISKTS]), y)
            cur_year = y
        lo = int(((int(ts[0]) // 60000) * 60000 - m0) // 60000)
        hi = min(len(mts), int((int(ts[-1]) + 59999 - m0) // 60000) + 1)
        X.process_day(px, ts, qt, m0, lo, hi, closes, fts, frates, fmarks, cov, alloc, cand["u"],
                      float(c["leverage"]), c["initial_market_fee"], c["later_limit_fee"], c["tp_limit_fee"],
                      c["reset_threshold"], c["min_limit_price_ratio"], p_mode, p_pos, p_neg, p_slip, p_k, p_lat,
                      fs, ist, lvl[0], lvl[1], lvl[2], lvl[3], cf, ci, eq_ts, eq_v)
        final_mark = float(closes[hi - 1])
        now = time.perf_counter()
        if now - tl >= 30:
            pc = (di + 1) / len(dates) * 100
            print(f"  {pc:5.1f}% | {d} | {(now - t0) / 60:5.1f} min (reading {tio / 60:4.1f}) | "
                  f"ETA {(now - t0) / pc * (100 - pc) / 60:5.1f} min", flush=True)
            tl = now
    if cur_year is None:
        print("No trades processed"); sys.exit(4)
    for s in range(S):
        close_obs(s)
        if ist[s, X.OVERFLOW]:
            print(f"ERROR: cycle storage overflow in {SIMS[s][0]} - results incomplete, do not use"); sys.exit(5)
    print(f"Pass complete in {(time.perf_counter() - t0) / 60:.1f} min (reading {tio / 60:.1f})")

    # ---------------- summaries ----------------
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    for s, sim in enumerate(SIMS):
        name = sim[0]
        n = int(ist[s, X.NCYC])
        rows = [{"open_ts": int(ci[s, i, X.CI_OPEN]), "close_ts": int(ci[s, i, X.CI_CLOSE]),
                 "depth": int(ci[s, i, X.CI_DEPTH]), "liquidated": int(ci[s, i, X.CI_LIQ]),
                 "worst_unrealized": float(cf[s, i, X.CF_WORST]), "M": float(cf[s, i, X.CF_M]),
                 "C_wallet_after": float(cf[s, i, X.CF_C]), "entry_cost": float(cf[s, i, X.CF_COST]),
                 "qty": float(cf[s, i, X.CF_QTY])} for i in range(n)]
        closed = [r for r in rows if not r["liquidated"] and r["M"] > 0 and r["C_wallet_after"] > 0]
        mae_a = [100.0 * r["worst_unrealized"] / r["C_wallet_after"] for r in closed]
        mae_c = [100.0 * r["worst_unrealized"] / r["M"] for r in closed]
        util = [100.0 * r["M"] / r["C_wallet_after"] for r in closed]
        opened = ist[s, X.OPEN] == 1 and ist[s, X.TERM] == 0
        last_ts, first_ts = int(ist[s, X.LASTTS]), int(ist[s, X.FIRSTTS])
        holds = [max(0, r["close_ts"] - r["open_ts"]) / 86400000.0 for r in rows if not r["liquidated"]]
        open_hold = max(0, last_ts - int(ist[s, X.OPENTS])) / 86400000.0 if opened else 0.0
        if opened:
            holds.append(open_hold)
        elapsed = max(60000, last_ts - first_ts + 60000)
        span_days = max(1e-9, elapsed / 86400000.0)
        exit_net = 0.0 if ist[s, X.LIQ] else exit_net_at(s, final_mark)
        roi = 100.0 * (exit_net / c["initial_wallet"] - 1.0)
        yrs = elapsed / (365.2425 * 86400000.0)
        depth = [r["depth"] for r in rows if not r["liquidated"]]
        res[name] = {
            "mode": sim[1], "ticks_processed": int(ist[s, X.TRADES]), "liquidated": bool(ist[s, X.LIQ]),
            "execution_invalid": bool(ist[s, X.INVALID]),
            "exit_net_return_pct": roi,
            "annualized_exit_net_return_pct": 100.0 * ((exit_net / c["initial_wallet"]) ** (1.0 / yrs) - 1.0) if exit_net > 0 else -100.0,
            "exit_net_terminal_equity": exit_net, "final_wallet": float(fs[s, X.W]),
            "max_drawdown_pct": 100.0 * float(fs[s, X.MDD]),
            "min_liquidation_room_pct": 100.0 * float(fs[s, X.MINROOM]) if np.isfinite(fs[s, X.MINROOM]) else None,
            "cycles_started": int(ist[s, X.TPC]) + int(ist[s, X.LIQ]) + (1 if opened else 0),
            "cycles_completed_tp": int(ist[s, X.TPC]), "reset_count": int(ist[s, X.RESETS]),
            "fees_total": float(fs[s, X.FEES]), "funding_paid_signed": float(fs[s, X.FUND]),
            "slippage_cost": float(fs[s, X.SLIP]), "tp_tick_rounding_shortfall": float(fs[s, X.TPSHORT]),
            "turnover_notional": float(fs[s, X.TURN]), "max_step": int(ist[s, X.MAXSTEP]),
            "max_hold_days": max(float(fs[s, X.MAXHOLD]) / 86400000.0, max(holds) if holds else 0.0),
            "hold_cycles": len(holds),
            "hold_days_p50": float(np.percentile(holds, 50)) if holds else None,
            "hold_days_p90": float(np.percentile(holds, 90)) if holds else None,
            "hold_days_p99": float(np.percentile(holds, 99)) if holds else None,
            "hold_days_mean": float(np.mean(holds)) if holds else None,
            **{f"hold_over_{k}d": sum(1 for h in holds if h > k) for k in (7, 30, 90)},
            **{f"deep_share_{k}d": sum(h for h in holds if h > k) / span_days for k in (7, 30, 90)},
            "open_hold_days": open_hold, "ended_in_position": bool(opened),
            "mae_vs_account_pct": pct_stats(mae_a, ("mean", "median", "p5", "p1", "worst")),
            "mae_vs_capital_pct": pct_stats(mae_c, ("mean", "median", "p5", "p1", "worst")),
            "utilization_pct": pct_stats(util, ("mean", "median", "p95", "max")),
            "ladder_depth_cycles": {"1-5": sum(1 for d in depth if d <= 5), "6-10": sum(1 for d in depth if 6 <= d <= 10),
                                    "11-12": sum(1 for d in depth if d >= 11)},
            "ladder_depth_sample": len(depth), "flat_funding_records": int(ist[s, X.FLATFUND]),
            "first_trade_ts": first_ts, "last_ts": last_ts, "final_mark": final_mark,
            "invalid_at_ts": int(ist[s, X.INVALIDTS]) or None,
            "yearly": years[name],
        }
        with open(OUT / f"cycles_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(list(rows[0].keys()) if rows else ["empty"])
            for r in rows:
                w.writerow([repr(v) for v in r.values()])
        with open(OUT / f"equity_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["first_risk_obs_ts_ms", "marked_equity_usd"])
            for i in range(int(ist[s, X.NEQ])):
                w.writerow([int(eq_ts[s, i]), repr(float(eq_v[s, i]))])
    (OUT / "results_eth.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ---------------- comparison with the client's reference ----------------
    exp = json.loads((HO / "reference-results" / "expected_metrics.json").read_text(encoding="utf-8"))
    pub = {x["id"]: x for x in exp["published_exit_net_scenarios"]}
    comp: dict = {}
    for sid in ("baseline", "funding", "slip1", "slip2"):
        ref, ours = exp["raw_results"][sid], res[sid]
        fields = {}
        for k, rv in ref.items():
            if k in ("candidate_name", "u", "n", "days"):
                continue
            ov = ours.get(k)
            if isinstance(rv, dict):
                for kk, rvv in rv.items():
                    fields[f"{k}.{kk}"] = (ov or {}).get(kk), rvv
            else:
                fields[k] = ov, rv
        fields["published_final_wallet_usd(exit-net)"] = ours["exit_net_terminal_equity"], pub[sid]["final_wallet_usd"]
        out = {}
        for k, (o, r) in fields.items():
            if isinstance(r, bool) or isinstance(r, int):
                st = "MATCH" if o == r else "DIFF"
            elif r is None or o is None:
                st = "N/A"
            else:
                d = abs(float(o) - float(r)); rel = d / max(1e-12, abs(float(r)))
                st = "MATCH" if d <= 1e-6 or rel <= 1e-12 else ("CLOSE" if rel <= 1e-6 else "DIFF")
            out[k] = {"ours": o, "ref": r, "status": st}
        comp[sid] = out
    ycomp: dict = {}
    for sid, fname in (("baseline", "eth_baseline_yearly.json"), ("funding", "eth_u82_yearly.json")):
        ry = {y["year"]: y for y in json.loads((HO / "reference-results" / fname).read_text(encoding="utf-8"))["years"]}
        ycomp[sid] = {}
        for oy in res[sid]["yearly"]:
            r = ry.get(oy["year"], {})
            ycomp[sid][oy["year"]] = {k: {"ours": oy.get(k), "ref": r.get(k)} for k in
                                      ("exit_net_return_pct", "marked_return_pct", "start_wallet", "end_wallet", "profit_usd",
                                       "max_drawdown_pct", "min_liquidation_room_pct", "max_hold_days", "max_step",
                                       "fees_total", "funding_paid_signed", "turnover_notional", "slippage_cost")}
    (OUT / "comparison.json").write_text(json.dumps({"scenarios": comp, "yearly": ycomp}, indent=2), encoding="utf-8")

    print("\nKEY COMPARISON (ours vs client reference):")
    for sid, out in comp.items():
        cnt = {st: sum(1 for v in out.values() if v["status"] == st) for st in ("MATCH", "CLOSE", "DIFF", "N/A")}
        print(f"  {sid:9} {cnt}")
        for k in ("ticks_processed", "final_wallet", "published_final_wallet_usd(exit-net)", "cycles_completed_tp",
                  "reset_count", "max_drawdown_pct", "min_liquidation_room_pct", "fees_total"):
            v = out.get(k)
            if v:
                print(f"      {k:40} ours={v['ours']!r:>24}  ref={v['ref']!r:>24}  {v['status']}")
    print("\nPHASE B SENSITIVITIES (baseline costs):")
    b = res["baseline"]["exit_net_terminal_equity"]
    for name in ("F1_tick", "F2_vol1x", "F3_vol2x", "L1_lat1s"):
        r = res[name]
        print(f"  {name:9} exit-net={r['exit_net_terminal_equity']:>14,.2f} ({(r['exit_net_terminal_equity'] / b - 1) * 100:+7.2f}%)"
              f"  tp={r['cycles_completed_tp']:>6}  liq={int(r['liquidated'])}  mdd={r['max_drawdown_pct']:.2f}%"
              f"{'  INVALID' if r['execution_invalid'] else ''}")


if __name__ == "__main__":
    main()
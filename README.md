# Independent Verification: Blitz T-DCA ETHUSDT Perpetual Backtest

**Author:** Ranveer Verma ([github.com/ranveer9](https://github.com/ranveer9)) · **Version:** v1.0.0
**Commissioned by:** Blitz Trading. The work was commissioned and paid for; the method, results and conclusions are the author's own, including all limitations and negative findings.

This is a historical verification of a backtest. It is not a guarantee of live trading performance.

📄 **Full report:** [`eth_verification_report.pdf`](eth_verification_report.pdf) (public edition)

## Scope
Binance ETHUSDT USD-M perpetual, long-only T-DCA, one published preset, four cost scenarios; 2021-01-01 00:00 UTC to 2026-08-25 00:00 UTC (exclusive); 2,062 checksum-verified daily aggTrades archives, 2,963,456,806 trades (identical to the reference count). The sealed period from 2026-08-25 was not used.

## Phase A: reproduction (reproduced bit-for-bit, two explained residuals)

| Scenario | Realised wallet (USD) | Exit-net equity, this report | Published exit-net | TP cycles | Max drawdown | Fields |
|---|---|---|---|---|---|---|
| Baseline | 1,947,269.80 | 1,946,324.41 | 1,946,288.58 | 45,104 | 50.2373% | 45 exact, 4 explained, 0 unexplained |
| Funding | 1,869,120.13 | 1,868,258.24 | 1,868,218.43 | 45,154 | 36.2376% | 42 exact, 6 explained, 0 unexplained |
| Funding + 1 bp TP slip | 1,658,258.88 | 1,657,453.25 | 1,657,416.87 | 44,986 | 35.5039% | 42 exact, 6 explained, 0 unexplained |
| Funding + 2 bp TP slip | 1,493,446.95 | 1,493,125.13 | 1,493,065.57 | 45,023 | 34.9238% | 42 exact, 6 explained, 0 unexplained |

Every cycle count, reset, fee, slippage, drawdown, liquidation-room, MAE, utilisation, ladder-depth and completed-cycle holding statistic matches the reference exactly in all four scenarios, as do the baseline total turnover and the baseline and funding annual turnover; the baseline realised wallet is identical to the last digit. Two residuals are explained, not adjusted:
1. **Recovered funding data:** in the funding scenarios the wallet gap equals the funding-paid gap to ten decimals (the original funding files were deleted; 3,100 of 6,186 funding marks were backfilled).
2. **Boundary inconsistency in the reference (confirmed by the client):** the published exit-net figures were valued at the 2026-08-25 00:00 mark, one minute inside the sealed period. This report values at the 2026-08-24 23:59 close; corrected baseline exit-net terminal equity **1,946,324.41 USD**. Corrected exit-net equity, ROI and open-position hold time for all four scenarios at full float64 precision: `results/terminal_values_2359.csv`.

## Phase B: adequacy of the assumptions (key findings)
Reproducing the model does **not** validate it as a description of exchange execution.
- **Marks and liquidation:** risk uses 1-minute mark **closes applied from the start of each minute** (information not yet available at that time), observed only at events, with interpolated marks and a flat maintenance rate (no tiers). Closest approach to the proxy liquidation line: 13.49% of mark; account drawdown 50.24%. **Zero modelled liquidations do not prove exchange-level survival.**
- **Fills:** a print at or through the price is assumed to fill in full. **A trade-through does not guarantee queue clearance or a full fill.** The reference also inherits a float64 tick artefact (a trade exactly at the decimal TP price may not trigger).
- **Latency:** 14,897 of 45,104 baseline cycles lasted under one second (6,021 within the same millisecond), 20.2% of closed-cycle profit.
- **Resets:** compounding assumes automatic wallet resets that the live bot does not perform.

**Quantified sensitivities** (baseline costs; each changes one rule):

| Rule | Exit-net equity (USD) | vs reference rules | TP cycles | Max drawdown | Liquidated |
|---|---|---|---|---|---|
| F1 one-tick trade-through | 1,970,210.55 | +1.23% | 45,156 | 49.90% | no |
| F2 volume at/through >= 1x size | 1,135,467.07 | -41.66% | 35,378 | 47.98% | no |
| F3 volume at/through >= 2x size | 1,060,026.55 | -45.54% | 34,198 | 51.59% | no |
| L1 1-second latency | 797,559.09 | -59.02% | 30,545 | 52.08% | no |

aggTrade quantity is a proxy for executable volume; the queue ahead is assumed and fills are all-or-nothing.

## ⚠️ What this public repository does NOT contain
At the client's instruction the exact ladder distances, allocations, configuration values and all cycle-level records are **private** and omitted. `code/ethsim_generic.py` contains the full event logic but reads every setting from a private configuration file (structure in `code/config_template.json`). **This public package therefore does not enable full public reproduction of the results.** No material finding is withheld; all findings are in the report.

## Contents
```
eth_verification_report.pdf        public edition of the report
results/scenario_comparison.csv    every compared field, four scenarios: independent vs reference
results/annual_comparison.csv      annual fields (baseline, funding): independent vs reference
results/annual_results.csv         this report's annual results, all four scenarios
results/sensitivities.csv          F1/F2/F3/L1 results
results/terminal_values_2359.csv   corrected exit-net equity, ROI and open-position hold time (23:59 valuation), full precision
results/verification_summary.json  verdicts and residual checks
code/ethsim_generic.py             generic engine (settings loaded from the private configuration)
code/run_eth_generic.py            runner
code/downloader.py                 data retrieval with official checksum verification
code/config_template.json          structure of the required (private) configuration
requirements.txt                   Python 3.12 dependencies
```

## Data retrieval
`https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-YYYY-MM-DD.zip` (+ `.CHECKSUM`) for 2021-01-01 to 2026-08-24 (2,062 files). `code/downloader.py` takes a manifest CSV with the columns `utc_date`, `archive_url`, `checksum_url`, verifies each archive against its official checksum, extracts it and skips days already extracted.

## Related
- BTC verification by the same author: https://github.com/ranveer9/blitz-btc-tdca-verification

## License
Code: MIT (see `LICENSE`). Report: © 2026 Ranveer Verma. Blitz Trading may host the full report with attribution and quote it accurately under the agreed review terms.

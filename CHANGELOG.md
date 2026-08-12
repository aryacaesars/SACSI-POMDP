# Changelog

## 2026-08-11

- Locked Virtual Garden field capacity at `0.35` after validation against the documented non-RL sanity references. This value is shared by every controller.
- Locked the initial SAC family reward as `reward_v1`: target-band bonus, out-of-band distance penalty, irrigation penalty, and action-change penalty.
- Added observation clipping to `[-5, 5]` and gradient clipping at norm `10` after validation-only diagnostics showed extreme inputs and critic-gradient spikes. The reward, data split, action bounds, and benchmark data were not changed or accessed for tuning.
- Changed SAC target entropy from the common `-1` default to `-3` after validation-only diagnostics showed alpha rising from `0.05` toward `0.90`, excessive exploration, and irrigation above the convergence limit. Automatic entropy tuning remains enabled.
- Revised the SAC reward from `reward_v1` to `reward_v2` using a 2:1 deficit-to-surplus distance penalty. Validation-only baseline diagnostics showed approximately 39% unavoidable rainfall-driven surplus, while crop-water deficit was the controllable failure preventing the gate. Water and action-change penalties remain unchanged; benchmark 2025 was not accessed.
- Revised `reward_v2` to `reward_v3` with a 5:1 deficit-to-surplus distance penalty after the matched seed-11 validation pilot improved target occupancy only from 43.40% to 44.81%. This is the final planned reward tuning before checkpoint-selection work; benchmark 2025 remains unopened.
- Rejected `reward_v3` after its matched seed-11 validation target occupancy fell to 42.40%; restored `reward_v2` as the selected objective. Added periodic validation-only checkpoint selection to prevent a later, degraded policy from replacing the best episode.
- Moved deterministic validation inference to a CPU copy of the actor while keeping training on CUDA. Single-observation sequential inference was slower on GPU and caused seed 22 to hit the 30-minute process timeout; weights and validation protocol are unchanged.
- Added SAC + Forecast with an explicitly labelled `SF-20_h1_controlled_proxy`: next-hour precipitation and ET0 receive 20% multiplicative noise, temperature receives training-scale additive noise, and forecasts never cross year/split boundaries. It is not claimed as an archived operational forecast.
- Added SAC + LSTM using causal `24×8` sequences and Residual Recurrent Warm-Start (RRWS) from the matched SAC Basic seed. Current-state actor/critics are frozen; zero-initialized recurrent residual branches are trained and audited with zero/reverse/shuffle history interventions.
- The 100-episode RRWS seed-11 run selected episode 0, leaving residual norm and all memory intervention deltas at zero. Added a validation-only early-stopping pilot (10 context-only episodes, validation every 2 episodes) to capture the active pre-overfit recurrent phase; benchmark 2025 remains unopened.
- Added SACSI Full with current 8-D warm-start, causal history `24×8`, and SF-20 h1 forecast 3-D. History and forecast actor residuals are independently zero-initialized and audited through separate norms plus zero/reverse/no-context interventions.
- Added a resumable expanded-training orchestrator for 10 matched seeds across SAC Basic, SAC + Forecast, SAC + LSTM, and SACSI Full. Existing checkpoints are skipped, failures are retained, and status/log files are updated after every run.

## 2026-08-12

- Completed the matched 10-seed expansion for all four RL families on CUDA: 40 checkpoints and four 10-row validation registries. All 18 automated tests passed after artifact verification.
- Completed the locked retrospective 2025 benchmark over nine methods on identical 8,760-hour support. Exported 45 per-run metric rows, 40 checkpoint audit rows, per-run time-series logs, and the nine-method summary without using 2025 for tuning or checkpoint reselection.
- Completed Sprint 11 on all 10 locked SACSI seeds: nine context conditions, matched SF10/SF20/SF30 forecast perturbations, k6/k12/k24/k48 inference-window sensitivity, and per-seed 2x2 factorial interactions. Full SACSI exactly reproduces Sprint 10 and no 2025 retraining or checkpoint reselection was performed.
- Completed Sprint 12 matched-seed statistics for the pre-specified Time in Target endpoint: one-df repeated-measures factorial contrasts, three paired SACSI comparisons with Holm correction, Cohen's dz, 20,000-resample bootstrap intervals, exact sign-flip confirmation, and deterministic baselines retained only as non-inferential trajectory references.
- Replaced Dashboard V1 with the unified Sprint 13 dashboard: nine-method registry, single/2–4 method comparison, matched RL seed selection, enriched 2025 trajectories, target band, irrigation/rain/cumulative-water plots, ablation/robustness and statistics views, browser-side PNG export, and CSV/XLSX/JSON/ZIP bundles.
- Added an English/Bahasa Indonesia dashboard language selector with localized navigation, guidance, chart labels, table headers, statistics messaging, and export controls while preserving model names and scientific identifiers.

# Panda Motion Planning & Residual RL

A seven-DoF manipulation project spanning joint-space motion planning, real MuJoCo servo execution, model compensation, and controlled BC/PPO evaluation.

[中文](README.md) · [Reproduction scope](docs/REPRODUCIBILITY.md) · [Final report](docs/32_项目最终验收报告.md)

![Recorded planning and actual stochastic training](assets/videos/overview_preview.gif)

[80-second overview](assets/videos/project_overview.mp4) · [Actual stochastic training](assets/videos/phase3dc_actual_training.mp4) · [Frozen policy comparison](assets/videos/phase3dc_policy_comparison.mp4)

**Verified outcomes:** seven-dimensional RRT-Connect/Shortcut with independent fine checking and actual ctrl+mj_step execution; an archived case where geometric planning clearance exceeded5mm but dynamics violated it; approximately59.3% lower mean TCP tracking RMSE for bounded integral compensation versus M3 in this final mismatch suite.

## Frozen final evaluation

Two new geometries × three mismatches × eight controllers/policies =48 complete executions. Both PPO seeds are retained. Metrics are recomputed from SHA-verified original tracking CSVs; compact float64 evidence is included.

| Method | TCP trajectory RMSE (mm) | Stable arrival | 5 mm violations / forbidden contacts |
|---|---:|---:|---:|
| M3 | 1.154319 | 6/6 | 0/0 |
| Integral500 | 0.472903 | 6/6 | 0/0 |
| Integral20 | 0.469299 | 6/6 | 0/0 |
| BC85 | 0.664839 | 6/6 | 0/0 |
| A_seed42 | 0.661844 | 6/6 | 0/0 |
| A_seed123 | 0.694725 | 6/6 | 0/0 |
| B_seed42 | 0.469134 | 6/6 | 0/0 |
| B_seed123 | 0.469692 | 6/6 | 0/0 |


**Negative result:** BC-warm-start PPO did not consistently close the expert gap. Integral20+PPO produced tiny, seed-dependent changes, with no established extra benefit over the strong integral baseline. This is a finite pilot, not a significance claim or a general impossibility result for RL.

## Technical scope

Native joint/actuator discovery, FK/Jacobian/DLS IK, keyboard teleoperation;23 robot collision geoms and274 audited pairs;0.02rad search and0.005rad final discrete validation; bidirectional RRT and checked shortcutting; quintic time scaling;500Hz physics/safety and20Hz residual actions; nominal/perturbed parameter isolation; gravity/velocity/integral compensation; BC78/85 and compatible tanh-mean Gaussian PPO initialization. Full-state and RGB records distinguish Actual Training from deterministic evaluation.

## Quick start

Use Python3.11 and the pinned requirements. `python scripts/setup_release.py --check` inspects the active environment. A clean network installation was NOT executed in this packaging stage; the tested host used its existing same-version, read-only dependency fallback. See the reproducibility guide rather than assuming a fresh-install guarantee.

```powershell
# Use an activated Python3.11 environment with the pinned dependencies.
python scripts/portfolio.py doctor
python scripts/portfolio.py model
python scripts/portfolio.py plan
python scripts/portfolio.py compare
python scripts/portfolio.py policy
python scripts/portfolio.py training-replay
python scripts/portfolio.py evaluate --compact
python scripts/portfolio.py metrics --compact
python -m pytest -q
```


`plan` validates a saved path; `compare` runs two new real dynamics evaluations without learning; `evaluate --compact` recomputes frozen evidence, not new trials. `policy` loads/checks BC and PPO weights. GUI variants `plan --viewer`, `training-replay --viewer`, and legacy keyboard teleop were not manually re-accepted this round. The legacy78D evaluation watcher is incompatible with85D PPO; use this release's `execute` entry.

All required source dependencies are retained. Historical222-test records are archival evidence; the relocated project's default suite is a separately executed lightweight release suite. Full historical datasets/checkpoints remain in the local research archive, not in Git. Included compact evidence cannot replace every original physics signal.

For actual re-execution of the frozen protocol, use `python scripts/portfolio.py evaluate --live --limit 1 --check` for preflight, then remove `--check`. Limit is explicitly bounded1..48. This packaging stage did not rerun all48 tasks or any training.

## Limits and attribution

Small fixed layouts, two training seeds, bounded optimization; no real hardware, Sim-to-Real result, continuous collision guarantee, original RL algorithm claim or safety certification. A safe simulated run does not imply certified safety.

The Panda assets retain their Apache-2.0 license and upstream attribution at Menagerie commit `cda5236f98383f6fbaa4820c6b18dba892cd087f`. See THIRD_PARTY_NOTICES.md. Original source has no public redistribution license selected yet. This new repository must remain **Private** until separate user approval; the prior Reacher authorization does not apply.

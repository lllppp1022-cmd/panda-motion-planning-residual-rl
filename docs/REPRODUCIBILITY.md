# Reproducibility and large-file contract

## Actually exercised in this release

The portable CLI's model inspection, saved-path fine validation, M3/Integral20 two-run dynamics comparison, six-model load/predict, actual training RGB/state recovery, original48-case metric recomputation and compact-evidence recomputation are tested. The default release pytest suite is executed separately. See report32 for logs/counts. No PPO learning occurs in these commands.

## Installation template versus installation acceptance

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
python -m pip install -r requirements.txt
python scripts/setup_release.py --check
python scripts/portfolio.py doctor
```

The environment-creation mechanism and installed dependency/API checks are lightweight acceptance steps; a clean network download/reinstall of every pinned wheel was NOT performed. These exact research pins may require matching Python/platform wheels and index availability. Do not silently substitute versions and claim an exact reproduction. The validated host was Windows/Python3.11 with a read-only same-version learning-package fallback; doctor records package origins honestly. On another machine install the packages locally instead of copying that host-specific path.

## Small commands / meanings

- `python scripts/portfolio.py model`: actual MJCF load, IDs, home state and274 pairs.
- `python scripts/portfolio.py plan`: saved RRT path, independent0.005rad collision revalidation; no new planning benchmark.
- `python scripts/portfolio.py execute --method M3`: one new real ctrl+mj_step evaluation in runtime/.
- `python scripts/portfolio.py compare`: paired M3/Integral20 evaluation; does not replace the frozen final48 results.
- `python scripts/portfolio.py policy`: SHA-check and load six bundled weights; seven finite bounded outputs.
- `python scripts/portfolio.py execute --method B_seed42 --check`: safe preflight without an execution. Remove--check for one true evaluation.
- `python scripts/portfolio.py training-replay`: unpack a copied actual stochastic episode, validate full recorded state/RGB; no predict or environment stepping.
- `python scripts/portfolio.py evaluate --compact`: reconstruct all final metrics from original-CSV-derived float64 evidence and original source hashes; no new trials.
- `python scripts/portfolio.py metrics --compact`: regenerate CSVs and final_recomputed.png.
- `python -m pytest -q`: only the portable release suite. Historical222 tests are retained under tests/historical; most require the original full research archive and cannot be honestly claimed to run from this small clone.

GUI switches `plan --viewer`, `training-replay --viewer` and legacy `scripts/teleop_panda.py` are available but were not manually re-tested this packaging round. They are not substitutes for the actual headless/recorded evidence checks. The legacy watcher checkpoint-evaluation mode still assumes78D; use the portable execute entry for85D PPO.

## Included versus local-only

- Included: all core source and historical scripts/configs, bundled upstream model/licenses, BC78/85 best checkpoints, four fixed4096 PPO checkpoints,48 compact float64 tracking projections, historical summaries/reports, actual episode0 RGB/full integration state, source videos and80-second edit.
- Local-only/ignored:48 full original tracking CSV.gz copies (about371MB), entire multi-phase training archive, every older checkpoint, environment caches and nominal/physics MJB files. Their original hashes are indexed in release/original_archive_manifest.json. The original research directory is retained in place; this is a frozen inventory, NOT a second full backup.
- Default metrics automatically use original CSV.gz if locally present;--compact forces the clone-compatible projection. It preserves double-precision tracking errors, velocities, TCP deltas, gap, phase, contact and timestamps without quantization. It is not the complete control/physics log. Source-to-projection lineage is recorded per case.
- Historical JSON summaries have projected archive-relative paths; original originals remain unchanged locally. They are not silently retargeted as newly executed experiments.
- To reproduce every historical test/rollout, obtain the corresponding full local archive from the owner and validate its hashes; no public dataset/weight download URL is invented here.

No continuous collision guarantee or hardware safety certification follows from discrete MuJoCo validation.

## 最终发布补充验收

`evaluate --live --limit 1 --check`是冻结协议复执行的实际预检命令；移除--check将使用清单中的路径、失配条件、物理种子和方法执行。limit必须1..48。已做1个回合验收，没有声称重新执行全部48回合。Git attributes保留源文件/模型/证据的原字节，不对研究源码或模型进行换行格式化。

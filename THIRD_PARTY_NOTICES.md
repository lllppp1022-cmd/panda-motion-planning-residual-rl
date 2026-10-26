# Third-party attribution and scope

## Panda model assets

Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f/franka_emika_panda

The complete franka_emika_panda directory, including LICENSE, README, CHANGELOG and mesh assets, is retained. Its applicable license is Apache-2.0. The Menagerie root license is a multi-model license catalog; licenses of unrelated, unbundled models are not assigned to project code. Original attribution/author notices are preserved.

Project-derived XML in models/derived adds named collision geometries, a hand-fixed tool site and project scenes. Native joint limits, affine position-servo gains, gear and force ranges are preserved. File hashes and modifications are documented in historical provenance; source-copy hashes appear in release/copy_provenance.json.

## Runtime libraries (not vendored)

MuJoCo, NumPy, SciPy, Matplotlib, PyYAML, pytest, Pillow, imageio/imageio-ffmpeg, PyTorch, Gymnasium, Stable-Baselines3 and TensorBoard are installed as dependencies. Their own licenses/notices remain applicable; no license here supersedes them. The release does not redistribute package source trees, wheels or FFmpeg binaries. Runtime package versions/metadata are recorded by setup_release.py and doctor.

Original project source/checkpoints are separately addressed in LICENSE.md. No affiliation with or endorsement by Franka, DeepMind or library authors is claimed.

# Official Panda source and project-derived scenes

Upstream: https://github.com/google-deepmind/mujoco_menagerie, commit cda5236f98383f6fbaa4820c6b18dba892cd087f, franka_emika_panda, Apache-2.0. Preserve the upstream LICENSE and README.

- mujoco_menagerie/: sparse Git checkout, original files unchanged.
- upstream_backup/franka_emika_panda/: exact80-file backup, SHA-validated.
- derived/panda_phase1.xml: adds named geoms and a hand-fixed tool_site; keeps original position-servo gains/gear/force ranges.
- derived/phase1_scene.xml: project floor, target marker, sphere and cylinder obstacles.

See results/model_provenance.json for every original/derived SHA and all modifications. No Reacher model, log or environment was copied.

# Resume bullets — adapt to verified personal contribution

**Panda motion planning and model-compensated control | Python, MuJoCo, PyTorch**

AI-assisted personal research project. Specify your actual role; do not claim sole authorship or hardware experience without evidence.

- Built seven-DoF FK/Jacobian/DLS position IK and dynamics-driven joint/Cartesian teleoperation.
- Implemented joint-space RRT-Connect and collision-checked shortcuts, independent0.005rad validation and quintic time scaling for real simulated servo execution.
- Audited23 robot collision geometries and274 pairs; retained a5mm dynamic-clearance violation despite an apparently valid geometric path.
- Evaluated nominal/perturbed parameter isolation and gravity, velocity and bounded integral compensation. Integral20 reduced mean TCP tracking RMSE from1.154319mm to0.469299mm in the fixed six-case mismatch suite (approximately59.3%).
- Ran bounded BC/PPO comparisons with two training seeds,16384 PPO decisions and447208 training physics steps; reported that PPO did not consistently outperform the strong integral baseline.
- Linked actual stochastic-training RGB, integration states, actions, rewards and policy-update boundaries; kept archival222-test evidence separate from relocated release acceptance.

No claim of a novel RL algorithm, statistically significant superiority, real-robot deployment, Sim-to-Real or certified safety.

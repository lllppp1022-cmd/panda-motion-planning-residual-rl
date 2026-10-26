"""Use installed dependencies first, optional read-only same-version Python fallback."""
import importlib.util,sys
from pathlib import Path

def activate():
    fallback=Path(r'D:\miniconda\envs\mujoco_learn\Lib\site-packages')
    if importlib.util.find_spec('gymnasium') is None and fallback.exists():
        # Append, never shadow Panda's pinned MuJoCo/numpy/scipy installations.
        sys.path.append(str(fallback))

def provenance():
    activate()
    import torch,gymnasium,stable_baselines3,tensorboard,numpy,mujoco
    return dict(executable=sys.executable,read_only_shared_fallback=True,
        packages={m.__name__:dict(version=m.__version__,source=m.__file__) for m in (torch,gymnasium,stable_baselines3,tensorboard,numpy,mujoco)})

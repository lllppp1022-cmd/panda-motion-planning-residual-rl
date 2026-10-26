from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import mujoco
from src.robot.model_loader import load_model
from src.robot.panda_interface import PandaInterface
@pytest.fixture(scope='session')
def model():return load_model()[0]
@pytest.fixture
def robot(model):
    d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'));mujoco.mj_forward(model,d);return PandaInterface(model,d)

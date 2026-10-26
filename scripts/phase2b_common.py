"""Load immutable recorded scene parameters and create independent query objects."""
import yaml
from src.common import ROOT
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_cases import restore_scene


def load_scenes():return yaml.safe_load((ROOT/'configs/phase2b_scenes.yaml').read_text(encoding='utf-8'))


def scene_checker(scene):
    c=CollisionChecker();restore_scene(c,scene['scene']);return c

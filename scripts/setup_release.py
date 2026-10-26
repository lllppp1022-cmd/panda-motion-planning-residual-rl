"""Inspect the active dependency set without installing packages or training."""
from _bootstrap import ROOT
import argparse,importlib,sys,json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true',required=True);a=p.parse_args()
from src.rl.dependencies import activate
activate()
names=['mujoco','numpy','scipy','matplotlib','yaml','pytest','PIL','imageio','gymnasium','stable_baselines3','torch','tensorboard']
result={}
for name in names:
    module=importlib.import_module(name);result[name]=dict(version=getattr(module,'__version__','unknown'),source=module.__file__)
print(json.dumps(dict(python=sys.version,dependencies=result,clean_network_install_tested=False),indent=2))
(ROOT/'runtime').mkdir(exist_ok=True);(ROOT/'runtime/setup_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

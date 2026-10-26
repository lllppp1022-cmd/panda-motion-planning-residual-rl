from pathlib import Path
import sys,time,traceback,argparse,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.common import write_json
def parser(description):
    p=argparse.ArgumentParser(description=description);p.add_argument('--viewer',action='store_true');p.add_argument('--record',action='store_true');p.add_argument('--speed',type=float,default=1.);return p
def destination(name):
    p=ROOT/'results/runs'/f'{name}_{time.time_ns()}';p.mkdir(parents=True);return p
def publish(dest,files):
    for rel in files:
        p=dest/rel;target=ROOT/'results'/p.name;shutil.copy2(p,target)
def main_guard(function):
    try:function()
    except Exception as e:
        error=ROOT/'results/errors'/f'{time.time_ns()}.json';write_json(error,dict(error=repr(e),traceback=traceback.format_exc()));raise

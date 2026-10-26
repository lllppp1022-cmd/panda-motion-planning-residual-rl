"""Finite zero-action interface probe, no optimizer or PPO training."""
from _bootstrap import ROOT,destination,main_guard
import subprocess,sys,time
import numpy as np
from src.training_observer.observer import ObservedEnv,atomic_json
from scripts.phase3c_common import sources
from src.common import write_json


def main():
    dest=destination('phase3d_preview_fix');env=ObservedEnv(sources('train'),dest/'actual_training',mode='evaluation',record=True,render_training=True)
    proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/watch_ppo_training.py'),'--live-stream',str(env.dest)],stdout=(dest/'display.log').open('w'),stderr=subprocess.STDOUT)
    try:
        env.reset(seed=42)
        for _ in range(32):env.step(np.zeros(7))
    finally:env.active_decision=0;env.close()
    atomic_json(env.dest/'live_status.json',dict(status='INTERFACE PROBE COMPLETE / NO PPO TRAINING'))
    write_json(dest/'complete.json',dict(passed=not env.record_errors,errors=env.record_errors,actual_physics=env.total_physics_steps,
        optimizer_updates=0,interface_steps=32,gui_pid=proc.pid,source='actual zero-residual interface probe'))
    write_json(ROOT/'results/phase3d_preview_fix_latest.json',dict(run=dest));assert not env.record_errors;print('FIX VERIFIED',dest,flush=True)


if __name__=='__main__':main_guard(main)

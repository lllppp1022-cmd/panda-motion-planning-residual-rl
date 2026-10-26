"""Freeze previous releases and audit actual historical episode/policy associations."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import numpy as np
from src.common import sha,write_json,write_csv
from src.training_observer.replay import RecordedEpisode
from scripts.phase3c_common import read,rows


def main():
    dest=destination('phase3d_history');manifest={}
    for name in ('phase3b','phase3c'):
        for p in (ROOT/'results/runs').glob(name+'_*'):
            for f in p.rglob('*'):
                if f.is_file():manifest[str(f.relative_to(ROOT))]=sha(f)
    for folder in ('src','scripts','configs','docs','tests'):
        for f in (ROOT/folder).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and 'phase3d' not in f.name.lower() and 'training_observer' not in f.parts and f.name!='watch_ppo_training.py':manifest[str(f.relative_to(ROOT))]=sha(f)
    write_json(dest/'history_sha256.json',manifest);write_json(ROOT/'results/phase3d_history_latest.json',dict(run=dest))
    base=ROOT/'results/runs/phase3c_training_continue_1792662132222765100';records=[];associations=[]
    for name in ('R0_seed42','R0_seed123','R1_seed42','R1_seed123'):
        run=base/name;s=rows(run/'training_samples.csv');checkpoints=sorted((run/'checkpoints').glob('*.zip'));npzs=sorted(run.glob('rollout_*.npz'))
        assert len(s)==2048 and len(npzs)==8 and len(checkpoints)==10
        for ep in range(8):
            data=RecordedEpisode(run,ep);errors=[]
            for i in range(0,len(data.t),50):
                data.set_state(i);tcp=data.data.site_xpos[data.site];expected=np.array([float(data.t[i][f'tcp_actual_{j}']) for j in range(3)])
                errors.append(float(np.linalg.norm(tcp-expected)))
            assert max(errors)<1e-12
            versions=sorted({x['policy_version'] for x in data.actions.values()})
            records.append(dict(group=name,episode=ep,physics_rows=len(data.t),decisions=len(data.actions),policy_versions=versions,
                saved_arm_qpos=True,saved_arm_qvel=True,saved_ctrl_seven=True,saved_tcp=True,full_qpos=False,full_qvel=False,
                fingers_complete=False,original_rgb=False,max_tcp_reconstruction_error_m=max(errors),
                outcome=data.summary['failure_reason'] or 'success',fidelity=data.fidelity))
            for a in data.actions.values():associations.append(dict(group=name,episode=ep,decision=a['global_decision'],policy_version=a['policy_version'],start_physics_step=a['start_step'],end_physics_step=a['end_step'],reward=a['interval_reward']))
            data.close()
        print('AUDITED',name,flush=True)
    write_csv(dest/'historical_episode_audit.csv',records);write_csv(dest/'decision_policy_associations.csv',associations)
    write_json(dest/'complete.json',dict(groups=4,episodes=32,historical_files=len(manifest),records=records,
        limitation='Historical gripper states, full integration state, warmstart and RGB were not saved; visible arm/FK reconstructs, not exact full-state restoration.',
        historical_failures='No full training task failure; each stage has a deliberately budget-interrupted unfinished episode.'))
    print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)

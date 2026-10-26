from _bootstrap import destination,main_guard
import json
from pathlib import Path
from src.common import ROOT,sha,write_json
from src.control.validated_path import load_verified_path


def main():
    dest=destination('phase2d_freeze');b=load_verified_path('D_two_obstacles',42)
    files=set()
    for name in ['phase2b_acceptance.json','phase2c_acceptance.json']:
        old=json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
        hashes=old.get('audited_after_runs_sha256',old.get('source_sha256',{}))
        for p,h in hashes.items():
            if sha(ROOT/p)!=h:raise RuntimeError('Historical acceptance mismatch: '+p)
            files.add(ROOT/p)
        files.add(ROOT/'results'/name)
    for folder in ['src','configs','tests','docs']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('phase2d'))
    for p in (ROOT/'results').glob('phase2[bc]*'):
        if p.is_file():files.add(p)
    for name in ['phase2b_batch_latest.json','phase2c_batch_latest.json']:
        run=Path(json.loads((ROOT/'results'/name).read_text())['run'])
        files.update(p for p in run.rglob('*') if p.is_file())
    files.update([ROOT/'models/derived/panda_phase1.xml',ROOT/'models/derived/phase2a_scene.xml',
                  ROOT/'assets/videos/phase2b_rrt_planning_demo.mp4',ROOT/'assets/videos/phase2c_dynamic_execution.mp4'])
    write_json(dest/'historical_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    write_json(dest/'historical_D_bundle.json',b.evidence)
    write_json(dest/'study_config.json',dict(config_path='configs/phase2d_margin_study.yaml',
                sha256=sha(ROOT/'configs/phase2d_margin_study.yaml'),no_controller_or_scene_changes=True))
    write_json(ROOT/'results/phase2d_freeze_latest.json',dict(run=dest));print(dest,len(files),'historical files',flush=True)


if __name__=='__main__':main_guard(main)

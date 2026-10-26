from _bootstrap import destination,main_guard
import json
from pathlib import Path
from src.common import ROOT,sha,write_json
from src.control.validated_path import load_verified_path


def main():
    dest=destination('phase2e_freeze');base=load_verified_path('D_two_obstacles',42);files=set()
    for folder in ['src','configs','tests','docs']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'phase2e' not in p.name.lower())
    for p in (ROOT/'results').glob('phase2[abcd]*'):
        if p.is_file():files.add(p)
    for label in ['phase2b_batch_latest','phase2c_batch_latest','phase2d_planning_latest','phase2d_dynamic_latest','phase2d_video_latest']:
        pointer=ROOT/'results'/(label+'.json');run=Path(json.loads(pointer.read_text(encoding='utf-8'))['run'])
        files.update(p for p in run.rglob('*') if p.is_file())
    files.update(p for p in (ROOT/'assets/videos').glob('phase2[abcd]*') if p.is_file())
    files.update([ROOT/'models/derived/panda_phase1.xml',ROOT/'models/derived/phase2a_scene.xml',ROOT/'results/model_provenance.json'])
    for p,h in json.loads((ROOT/'results/phase2d_acceptance.json').read_text(encoding='utf-8'))['hashes'].items():
        if sha(ROOT/p)!=h:
            fixpath=ROOT/'results/phase2e_test_artifact_fix.json'
            if p!='tests/test_phase2d_margin.py' or not fixpath.exists():raise ValueError('Phase2D input changed: '+p)
            fix=json.loads(fixpath.read_text(encoding='utf-8'))
            if sha(fix['original_backup'])!=h or sha(ROOT/p)!=fix['patched_sha256']:raise ValueError('Unverified regression source fix')
    write_json(dest/'historical_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    write_json(dest/'D_historical_bundle.json',base.evidence)
    write_json(ROOT/'results/phase2e_freeze_latest.json',dict(run=dest));print(dest,len(files),flush=True)


if __name__=='__main__':main_guard(main)

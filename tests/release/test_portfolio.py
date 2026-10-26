from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.common import ROOT,sha
from src.planning.collision_checker import CollisionChecker
from src.control.trajectory_generator import QuinticTrajectory
from scripts.portfolio import manifest,checkpoint,selected_scene

def test_native_home_model_and_all_pairs():
    c=CollisionChecker();assert (c.model.nq,c.model.nv,c.model.nu)==(9,9,8);assert len(c.pairs)==274 and c.state_valid(c.home).valid
def test_bad_state_inputs_and_limits():
    c=CollisionChecker()
    for q in (np.zeros(6),np.full(7,np.nan),np.full(7,np.inf),np.full(7,20)):assert not c.state_valid(q).valid
def test_saved_final_paths_remain_fine_validated():
    from scripts.run_phase3a import load_new
    for scene in manifest('scenes.json'):
        if scene['split']=='test':assert load_new(ROOT/scene['path']).path.shape[1]==7
def test_checkpoints_all_match_release_hashes():
    for name in manifest('models.json'):assert checkpoint(name).is_file()
def test_48_original_tracking_archives():
    rr=manifest('final_cases.json');assert len(rr)==48
    assert all(sha(ROOT/r['numeric_evidence'])==r['numeric_sha256'] for r in rr)
def test_recomputed_data_not_hardcoded_results():
    import csv
    with (ROOT/'results/summary/final_cases_recomputed.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
    assert len(rr)==48 and all(r['success']=='True' for r in rr)
def test_third_party_license_retained():
    text=(ROOT/'models/mujoco_menagerie/franka_emika_panda/LICENSE').read_text()
    assert 'Apache License' in text and (ROOT/'THIRD_PARTY_NOTICES.md').is_file()
def test_no_training_in_portable_default_entry():
    assert '.learn(' not in (ROOT/'scripts/portfolio.py').read_text(encoding='utf-8')

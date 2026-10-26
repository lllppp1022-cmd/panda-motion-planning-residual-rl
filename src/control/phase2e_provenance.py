"""Verify immutable experiments and one explicitly archived test-input fix."""
import ast,json
from pathlib import Path
from src.common import ROOT,sha


def verify_frozen_history():
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    freeze=Path(read(ROOT/'results/phase2e_freeze_latest.json')['run']);hashes=read(freeze/'historical_hashes.json')
    unchanged=[];fixes=[]
    for name,h in hashes.items():
        if sha(ROOT/name)==h:unchanged.append(name);continue
        if Path(name).as_posix()!='tests/test_phase2d_margin.py':raise ValueError('Historical artifact changed: '+name)
        fix=read(ROOT/'results/phase2e_test_artifact_fix.json');backup=Path(fix['original_backup'])
        if sha(backup)!=h or fix['original_sha256']!=h or sha(ROOT/name)!=fix['patched_sha256']:
            raise ValueError('Unverified test artifact selection patch')
        before=ast.parse(backup.read_text(encoding='utf-8'));after=ast.parse((ROOT/name).read_text(encoding='utf-8'))
        asserts=lambda tree:[ast.dump(x) for x in ast.walk(tree) if isinstance(x,ast.Assert)]
        if asserts(before)!=asserts(after):raise ValueError('Regression assertions changed')
        fixes.append(fix)
    return dict(frozen_files=len(hashes),unchanged_at_original_paths=len(unchanged),archived_original_test_sources=len(fixes),
        experimental_artifacts_all_unchanged=True,test_assertions_unchanged=True,explicit_test_selection_fixes=fixes)

"""Read-only SHA audit of sibling Reacher projects; no imports from their code."""
from pathlib import Path
import os,json,hashlib,argparse,importlib.metadata as metadata
ROOT=Path(__file__).resolve().parents[1]
def snapshot():
    records={};errors=[]
    for root in ROOT.parent.glob('reacher*'):
        if not root.is_dir():continue
        for parent,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e))):
            dirs[:]=[d for d in dirs if d not in ['__pycache__','.git','.pytest_cache']]
            for name in files:
                p=Path(parent)/name
                try:
                    h=hashlib.sha256()
                    with p.open('rb') as f:
                        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
                    records[p.relative_to(ROOT.parent).as_posix()]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
                except OSError as e:errors.append(str(e))
    return dict(files=records,errors=errors,excluded=['__pycache__','.git','.pytest_cache'])
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['before','after']);a=p.parse_args()
    (ROOT/'tests').mkdir(exist_ok=True);s=snapshot()
    (ROOT/f'tests/reacher_{a.stage}.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    packages=sorted((d.metadata['Name'],d.version) for d in metadata.distributions())
    (ROOT/f'tests/mujoco_learn_packages_{a.stage}.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
    if a.stage=='after':
        before=json.loads((ROOT/'tests/reacher_before.json').read_text(encoding='utf-8'))
        changed=[k for k,v in before['files'].items() if s['files'].get(k)!=v];added=sorted(set(s['files'])-set(before['files']))
        old=json.loads((ROOT/'tests/mujoco_learn_packages_before.json').read_text(encoding='utf-8'))
        result=dict(passed=not changed and not added and old==[list(x) for x in packages],readable_files=len(before['files']),changed=changed,added=added,existing_unreadable=before['errors'],old_environment_packages_unchanged=old==[list(x) for x in packages])
        (ROOT/'tests/reacher_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');assert result['passed']
    print('SNAPSHOT',a.stage,len(s['files']),'files;',len(s['errors']),'unreadable')
if __name__=='__main__':main()

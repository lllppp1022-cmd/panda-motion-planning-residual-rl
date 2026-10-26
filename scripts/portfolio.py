"""Portable release entry: archived evidence or bounded real evaluation; NO training."""
if __package__:
    from scripts._bootstrap import ROOT,main_guard
else:
    from _bootstrap import ROOT,main_guard
import argparse,json,hashlib,gzip,csv,io,time,sys,shutil,subprocess
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha

def manifest(n):return json.loads((ROOT/'release'/n).read_text(encoding='utf-8'))
def runtime(name):
    p=ROOT/'runtime/runs'/f'{name}_{time.time_ns()}';p.mkdir(parents=True);return p
def checkpoint(name):
    m=manifest('models.json')[name];p=ROOT/m['path']
    if sha(p)!=m['sha256']:raise ValueError('Checkpoint hash mismatch: '+name)
    return p
def selected_scene(name=None):
    choices=manifest('scenes.json');r=next((s for s in choices if s['scene']==name),None) if name else next(s for s in choices if s['split']=='test')
    if r is None:raise ValueError('Unknown scene')
    return ROOT/r['path']

def run_controller(source,condition,seed,method,folder):
    from src.rl.dependencies import activate
    activate()
    from src.bc_residual.runner import run_case
    from src.phase3dc.common import execute
    from stable_baselines3 import PPO
    if method in ('M3','Integral20','Integral500','BC85'):
        return run_case((str(source),condition,seed,method,str(checkpoint('BC85')) if method=='BC85' else None,str(folder)))
    return execute(source,condition,seed,method[0],folder,policy=PPO.load(checkpoint(method),device='cpu'))

def metrics(compact=False):
    cases=manifest('final_cases.json');detail=[]
    for case in cases:
        p=ROOT/case['tracking_gzip']
        if compact or not p.exists():
            numeric=ROOT/case['numeric_evidence'];assert sha(numeric)==case['numeric_sha256'];z=np.load(numeric)
            assert str(z['original_csv_sha256'])==case['original_csv_sha256'];mask=np.isin(z['phase'],[1,2]);tcp=np.linalg.norm(z['tcp_delta'],axis=1)
            good=np.isin(z['phase'],[2,3])&(np.max(abs(z['q_error']),axis=1)<=.02)&(tcp<=.01)&(np.max(abs(z['qvel']),axis=1)<=.02)
            stable=0.
            for value in good:stable=stable+.002 if value else 0.
            violation=int(np.sum(~z['valid']));contact=int(np.sum(z['contact']));success=stable>=.5-1e-9 and int(z['phase'][-1]) in (2,3) and violation==contact==0
            assert success==case['archived_success']
            detail.append(dict(case=case['case'],scene=case['scene'],condition=case['condition']['name'],method=case['method'],success=success,
                tcp_rmse_mm=float(np.sqrt(np.mean(tcp[mask]**2))*1000),joint_rmse_rad=float(np.sqrt(np.mean(z['q_error'][mask]**2))),final_tcp_mm=float(tcp[-1]*1000),
                minimum_clearance_mm=float(np.min(z['gap'])*1000),safety_violation_steps=violation,prohibited_contact_observations=contact,physics_steps=len(tcp),source_csv_sha256=case['original_csv_sha256']))
            z.close();continue
        raw=gzip.decompress(p.read_bytes());assert hashlib.sha256(raw).hexdigest()==case['original_csv_sha256']
        ssq=0.;qssq=0.;n=0;gap=float('inf');violation=0;contact=0;stable=0.;physics=0;lasttcp=0.
        for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
            physics+=1
            qa=np.array([float(row[f'q_actual_{j}']) for j in range(1,8)]);qr=np.array([float(row[f'q_ref_{j}']) for j in range(1,8)])
            va=np.array([float(row[f'dq_actual_{j}']) for j in range(1,8)])
            tcp=np.array([float(row[f'tcp_actual_{j}'])-float(row[f'tcp_ref_{j}']) for j in range(3)])
            lasttcp=float(np.linalg.norm(tcp));assert abs(lasttcp-float(row['tcp_error_m']))<1e-10
            g=float(row['minimum_clearance_m']);gap=min(gap,g);violation+=int(row['safety_valid']!='True');contact+=int(row['prohibited_contact_count'])
            if row['phase'] in ('trajectory','settle'):ssq+=lasttcp**2;qssq+=float(np.sum((qa-qr)**2));n+=1
            good=row['phase'] in ('settle','completed') and np.max(abs(qa-qr))<=.02 and lasttcp<=.01 and np.max(abs(va))<=.02
            stable=stable+.002 if good else 0.
        success=stable>=.5-1e-9 and row['phase'] in ('settle','completed') and violation==contact==0
        assert success==case['archived_success']
        detail.append(dict(case=case['case'],scene=case['scene'],condition=case['condition']['name'],method=case['method'],success=success,
            tcp_rmse_mm=np.sqrt(ssq/n)*1000,joint_rmse_rad=np.sqrt(qssq/(n*7)),final_tcp_mm=lasttcp*1000,minimum_clearance_mm=gap*1000,
            safety_violation_steps=violation,prohibited_contact_observations=contact,physics_steps=physics,source_csv_sha256=case['original_csv_sha256']))
    summary=[]
    for method in dict.fromkeys(r['method'] for r in detail):
        rr=[r for r in detail if r['method']==method]
        summary.append(dict(method=method,cases=len(rr),successes=sum(r['success'] for r in rr),tcp_rmse_mm=float(np.mean([r['tcp_rmse_mm'] for r in rr])),
            joint_rmse_rad=float(np.mean([r['joint_rmse_rad'] for r in rr])),minimum_clearance_mm=min(r['minimum_clearance_mm'] for r in rr),
            safety_violation_steps=sum(r['safety_violation_steps'] for r in rr),prohibited_contacts=sum(r['prohibited_contact_observations'] for r in rr)))
    output=ROOT/'results/summary';write_csv(output/'final_cases_recomputed.csv',detail);write_csv(output/'final_controller_metrics.csv',summary)
    write_json(output/'metric_validation.json',dict(cases=len(detail),source='SHA checked original CSV or lossless float64 error projection; TCP/joint RMSE independently recomputed',compact_requested=compact,success_derived_from_unchanged_thresholds=True,new_experiments=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    ax.bar([r['method'] for r in summary],[r['tcp_rmse_mm'] for r in summary],color='#357da2',alpha=.7)
    for j,r in enumerate(summary):ax.scatter(j+np.linspace(-.1,.1,6),[x['tcp_rmse_mm'] for x in detail if x['method']==r['method']],c='black',s=20)
    ax.set(ylabel='TCP trajectory RMSE (mm)',title='Frozen final evaluation: 2 geometries x 3 mismatches / all seeds');ax.tick_params(axis='x',rotation=20)
    fig.savefig(ROOT/'assets/plots/final_recomputed.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2));return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['doctor','model','plan','execute','compare','policy','training-replay','evaluate','metrics','media'])
    p.add_argument('--method',default='M3',choices=['M3','Integral20','Integral500','BC85','A_seed42','A_seed123','B_seed42','B_seed123'])
    p.add_argument('--scene');p.add_argument('--viewer',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--compact',action='store_true');p.add_argument('--live',action='store_true');p.add_argument('--limit',type=int,default=48);a=p.parse_args()
    import os
    os.chdir(ROOT)
    if a.command=='evaluate' and a.live:
        if not 1<=a.limit<=48:raise ValueError('Bounded evaluation limit1..48 required')
        folder=runtime('frozen_live_evaluation');results=[]
        from src.rl.dependencies import activate
        activate()
        from scripts.run_phase3a import load_new
        for case in manifest('final_cases.json')[:a.limit]:
            source=ROOT/case['path'];load_new(source)
            if case['method'].startswith(('A_','B_','BC')):checkpoint(case['method'])
            if a.check:print('FROZEN LIVE PREFLIGHT',case['case']);continue
            result=run_controller(source,case['condition'],case['physical_seed'],case['method'],folder/case['case']);results.append(result)
        write_json(folder/'complete.json',dict(results=results,preflight_only=a.check,requested_limit=a.limit,training=False,historical_results_unchanged=True));print('OUTPUT',folder);return
    if a.command in ('metrics','evaluate'):metrics(a.compact);return
    if a.command=='doctor':
        from src.rl.dependencies import activate,provenance
        activate();r=provenance();r['same_version_fallback_is_not_a_clean_install']=any('mujoco_learn' in v['source'] for v in r['packages'].values())
        write_json(ROOT/'runtime/doctor.json',r);print(json.dumps(r,indent=2));return
    if a.command=='media':
        import imageio.v2 as im
        r={}
        for name in ('project_overview.mp4','phase3dc_actual_training.mp4','phase3dc_policy_comparison.mp4'):
            path=ROOT/'assets/videos'/name
            with im.get_reader(str(path)) as reader:
                metadata=reader.get_meta_data();metadata.pop('nframes',None) # ffmpeg reports infinity for streaming containers
                r[name]=dict(metadata=metadata,decoded_frames=sum(1 for _ in reader))
        write_json(ROOT/'runtime/media_validation.json',r);print(json.dumps(r,indent=2));return
    from src.rl.dependencies import activate
    activate()
    if a.command=='policy':
        import torch
        torch.set_num_threads(1)
        from src.bc_residual.core import Policy
        from stable_baselines3 import PPO
        for name in manifest('models.json'):
            model=Policy(checkpoint(name)) if name.startswith('BC') else PPO.load(checkpoint(name),device='cpu')
            out=model.predict(np.zeros(85,dtype=np.float32)) if name.startswith('BC') else model.predict(np.zeros(85,dtype=np.float32),deterministic=True)[0]
            assert out.shape==(7,) and np.isfinite(out).all() and np.max(abs(out))<=1
            print(name,'LOAD + bounded7D predict OK')
        return
    if a.command=='training-replay':
        source=ROOT/'evidence/actual_training/A_seed42';dest=runtime('actual_training_replay')/'A_seed42';shutil.copytree(source,dest/'actual_training')
        shutil.move(str(dest/'actual_training/training_samples.csv'),str(dest/'training_samples.csv'))
        for f in (dest/'actual_training').rglob('*.gz'):
            with gzip.open(f,'rb') as i,f.with_suffix('').open('wb') as o:shutil.copyfileobj(i,o)
        from src.training_observer.replay import RecordedEpisode
        data=RecordedEpisode(dest,0,'actual');assert data.fidelity=='TRAINING-SITE RGB';assert data.image(100).shape==(832,960,3);data.close()
        print('Actual Training RGB/state verified; replay folder',dest)
        if a.viewer:subprocess.run([sys.executable,str(ROOT/'scripts/watch_ppo_training.py'),'--mode','actual','--run',str(dest),'--episode','0','--paused'],check=True)
        return
    from src.planning.collision_checker import CollisionChecker
    if a.command=='model':
        c=CollisionChecker();r=c.state_valid(c.home);print('nq/nv/nu',c.model.nq,c.model.nv,c.model.nu,'home valid',r.valid,'pairs',len(c.pairs));assert r.valid;return
    source=selected_scene(a.scene)
    from scripts.run_phase3a import load_new
    bundle=load_new(source)
    if a.command=='plan':
        print('KINEMATIC PLANNING PLAYBACK / NOT DYNAMIC SERVO EXECUTION');print('saved path nodes',len(bundle.path),'independent0.005rad verification OK')
        if a.viewer:
            import mujoco,mujoco.viewer
            c=bundle.checker
            with mujoco.viewer.launch_passive(c.model,c.data) as viewer:
                for start,end in zip(bundle.path,bundle.path[1:]):
                    for t in np.linspace(0,1,60):
                        if not viewer.is_running():return
                        c.data.qpos[c.robot.qpos_indices]=start+(end-start)*t;mujoco.mj_forward(c.model,c.data);viewer.sync();time.sleep(.02)
        return
    if a.command in ('execute','compare'):
        folder=runtime(a.command);condition={'name':'gain86','kp_scale':.86}
        methods=['M3','Integral20'] if a.command=='compare' else [a.method];results=[]
        if a.check:
            for method in methods:
                if method.startswith(('A_','B_','BC')):checkpoint(method)
            print('CONFIG/PATH/CHECKPOINT preflight OK; no physical execution');return
        from src.bc_residual.runner import run_case
        from src.phase3dc.common import execute
        from stable_baselines3 import PPO
        for method in methods:
            if method in ('M3','Integral20','Integral500','BC85'):r=run_case((str(source),condition,8201,method,str(checkpoint('BC85')) if method=='BC85' else None,str(folder/method)))
            else:r=execute(source,condition,8201,method[0],folder/method,policy=PPO.load(checkpoint(method),device='cpu'))
            results.append(r);print(method,r['success'],r['tcp_rmse_m']*1000,'mm',r['minimum_actual_clearance_m']*1000,'mm min gap')
        write_json(folder/'complete.json',dict(source=str(source),method_results=results,actual_dynamic_execution=True,not_training=True));print('OUTPUT',folder)

if __name__=='__main__':main_guard(main)

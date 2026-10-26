from _bootstrap import ROOT,destination,main_guard
from src.rl.dependencies import activate
activate()
import argparse,concurrent.futures
import torch
from stable_baselines3 import PPO
from src.phase3dc.common import *
from src.phase3dc.environment import design
from src.bc_residual.runner import run_case


def job(args):
    source,condition,seed,name,checkpoint,folder,record=args;torch.set_num_threads(1)
    if name in ('M3','Integral500','Integral20','BC85'):
        r=run_case((str(source),condition,seed,name,bc_checkpoint() if name=='BC85' else None,str(folder)),record_video=record)
    else:
        model=PPO.load(checkpoint,device='cpu');r=execute(source,condition,seed,name[0],folder,policy=model,record=record)
    r.update(evaluation_method=name,checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint) if checkpoint else '')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['baseline','gate','post','final'],required=True);a=p.parse_args();d=destination('phase3dc_evaluation_'+a.stage)
    cfg=design();split='test' if a.stage=='final' else 'validation';conditions=cfg['test_conditions'] if split=='test' else cfg['validation_conditions'];seed=cfg['test_seed'] if split=='test' else cfg['validation_seed']
    methods=[(m,None) for m in ('M3','Integral500','Integral20','BC85')];training=None
    if a.stage!='baseline':
        if a.stage=='gate':training=Path(read(ROOT/'results/phase3dc_training_initial_latest.json')['run'])
        else:
            gate=read(ROOT/'results/phase3dc_2048_gate.json');label='continue' if gate['all_four_passed'] else 'initial'
            training=Path(read(ROOT/f'results/phase3dc_training_{label}_latest.json')['run'])
        runs=read(training/'complete.json')['results'];methods+= [(r['method']+'_seed'+str(r['seed']),r['checkpoint']) for r in runs]
        if a.stage in ('gate','post'):methods=methods[4:]
    if a.stage=='final':write_json(d/'final_test_opening.json',dict(protocol=cfg,code_hashes=code_hashes(),training_run=training,interpretation=read(ROOT/'configs/phase3dc_interpretation.json'),checkpoints=[dict(method=m,path=c,sha256=sha(c) if c else '') for m,c in methods],no_more_selection=True))
    jobs=[(s,c,seed,name,ck,str(d/f'{s.stem}_{c["name"]}_{name}'),False) for s in sources(split) for c in conditions for name,ck in methods]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(job,jobs):results.append(r);print('EVALUATED',r['evaluation_method'],r['condition']['name'],r['success'],r['tcp_rmse_m']*1000,flush=True)
    write_json(d/'complete.json',dict(results=results,stage=a.stage,split=split,training_run=training));write_json(ROOT/f'results/phase3dc_evaluation_{a.stage}_latest.json',dict(run=d))
    if a.stage=='gate':gate_report(training,results)


def gate_report(training,evaluations):
    cfg=design();baseline=read(Path(read(ROOT/'results/phase3dc_evaluation_baseline_latest.json')['run'])/'complete.json')['results'];group=[]
    for r in read(training/'complete.json')['results']:
        name=r['method']+'_seed'+str(r['seed']);ev=[e for e in evaluations if e['evaluation_method']==name];reference='BC85' if r['method']=='A' else 'Integral20'
        base=[e for e in baseline if e['evaluation_method']==reference];mean=float(np.mean([e['tcp_rmse_m'] for e in ev]));initial=float(np.mean([e['tcp_rmse_m'] for e in base]))
        samples=csvrows(Path(r['run_directory'])/'training_samples.csv');effective=np.array([json.loads(s['effective_correction']) for s in samples]);requested=np.array([json.loads(s['correction_target']) for s in samples]);mask=abs(requested)>1e-12
        fraction=float(np.mean(abs(effective[mask])>1e-8));mismatch=sum(e['success'] and e['condition']!='nominal' for e in r['completed'])
        checks=dict(training_safe=not any(e['safety_violation_steps'] or e['prohibited_contact_observations'] for e in r['episode_results']),
            validation_safe_success=all(e['success'] and not e['safety_violation_steps'] and not e['prohibited_contact_observations'] for e in ev),
            updated=all(e['actor_parameter_delta_l2']>0 and e['critic_parameter_delta_l2']>0 for e in r['updates']),
            finite=all(e[k] is not None and np.isfinite(e[k]) for e in r['updates'] for k in ('value_loss','approx_kl','policy_gradient_loss')),
            enough_mismatch=mismatch>=cfg['gate']['minimum_mismatched_completed_episodes'],
            effective=fraction>=cfg['gate']['minimum_effective_correction_fraction'],
            no_unacceptable_degradation=mean<=initial*(1+cfg['gate']['max_relative_mean_tcp_degradation']))
        group.append(dict(method=r['method'],seed=r['seed'],checks=checks,passed=all(checks.values()),validation_tcp_mm=mean*1000,initial_tcp_mm=initial*1000,effective_fraction=fraction,mismatched_completed=mismatch))
    write_json(ROOT/'results/phase3dc_2048_gate.json',dict(training_run=str(training),all_four_passed=all(g['passed'] for g in group),groups=group,rule=cfg['gate']))
    print('COMMON GATE',group,flush=True)

if __name__=='__main__':main_guard(main)

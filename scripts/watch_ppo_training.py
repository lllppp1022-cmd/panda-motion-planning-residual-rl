"""Explicit Actual Training / Checkpoint Evaluation / Baseline Comparison player."""
from _bootstrap import ROOT,destination,main_guard
import argparse,json,time,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageTk
from src.training_observer.replay import RecordedEpisode
from src.training_observer.observer import atomic_json
from scripts.phase3c_common import read,sources,evaluate_job
from src.common import write_json


def gui(data=None,live=None,session=None,speed=1.,paused=False,decision=None):
    import tkinter as tk
    root=tk.Tk();root.title('Panda PPO | '+('Actual Training LIVE RGB' if live else data.lines(0)[0]));events=[]
    canvas=tk.Label(root,bg='#132133');canvas.pack();status=tk.StringVar(value='Waiting for actual training frames...' if live else data.lines(0)[0]);tk.Label(root,textvariable=status,font=('Consolas',11)).pack()
    panel=tk.Frame(root);panel.pack();state=dict(index=0,paused=paused,speed=speed,wall=time.perf_counter(),last_image=None,drawn=-1)
    if data and decision is not None:
        if decision not in data.actions:raise ValueError('Decision not present in selected episode')
        state['index']=data.actions[decision]['end_step']-1;data.set_rgb_mode(False)
    def record(action):
        events.append(dict(action=action,index=state['index'],paused=state['paused'],speed=state['speed'],wall=time.time()));write_json(session/'gui_events.json',events)
    def toggle():state['paused']=not state['paused'];record('pause_toggle')
    def move(amount,decision=False):
        if live:return
        state['paused']=True
        if decision:
            data.set_rgb_mode(False)
            candidates=[x-1 for x in data.end_steps]
            choices=[x for x in candidates if x>state['index']] if amount>0 else [x for x in candidates if x<state['index']]
            state['index']=(min(choices) if amount>0 else max(choices)) if choices else state['index']
        else:state['index']=int(np.clip(state['index']+amount,0,len(data.t)-1))
        record('decision_jump' if decision else 'frame_jump')
    def rate(factor):state['speed']=float(np.clip(state['speed']*factor,.125,8));record('speed_change')
    def view_mode():
        if not live:data.set_rgb_mode(not data.use_rgb);state['drawn']=-1;record('rgb_state_mode_switch')
    def close():record('window_close_only');root.destroy()
    def abort():
        (live/'abort_training.request').write_text('USER EXPLICITLY REQUESTED TRAINING ABORT',encoding='utf-8');record('explicit_training_abort')
    tk.Button(panel,text='Pause / Continue (P)',command=toggle).pack(side='left')
    if not live:
        for label,callback in [('Previous decision B',lambda:move(-1,True)),('Next decision N',lambda:move(1,True)),('Frame -',lambda:move(-1)),('Frame +',lambda:move(1))]:tk.Button(panel,text=label,command=callback).pack(side='left')
    else:tk.Button(panel,text='ABORT TRAINING (restricted hold)',command=abort,bg='#ffbbbb').pack(side='left')
    if not live:tk.Button(panel,text='RGB / State (V)',command=view_mode).pack(side='left');root.bind('<v>',lambda event:view_mode())
    tk.Button(panel,text='Slower -',command=lambda:rate(.5)).pack(side='left');tk.Button(panel,text='Faster +',command=lambda:rate(2)).pack(side='left')
    for key,fn in [('p',toggle),('space',toggle),('n',lambda:move(1,True)),('b',lambda:move(-1,True)),('Left',lambda:move(-1)),('Right',lambda:move(1)),('plus',lambda:rate(2)),('equal',lambda:rate(2)),('minus',lambda:rate(.5)),('Escape',close),('q',close)]:root.bind('<'+key+'>',lambda event,fn=fn:fn())
    root.protocol('WM_DELETE_WINDOW',close)
    def tick():
        now=time.perf_counter();elapsed=now-state['wall'];state['wall']=now
        try:
            if live:
                s=read(live/'live_status.json') if (live/'live_status.json').exists() else {};status.set(str({k:s[k] for k in ('status','episode','global_decision','policy_version') if k in s})+' | X/Q closes display only')
                path=Path(s['rgb_path']) if s.get('rgb_path') else live/'live_frame.png'
                if path.exists() and not state['paused']:
                    with Image.open(path) as im:photo=ImageTk.PhotoImage(im.resize((864,749)))
                    canvas.configure(image=photo);state['last_image']=photo
            else:
                if not state['paused']:state['index']=min(len(data.t)-1,state['index']+max(0,int(elapsed*state['speed']/.002)))
                if state['index']!=state['drawn']:
                    image=data.image(state['index']);photo=ImageTk.PhotoImage(Image.fromarray(image).resize((864,749)));canvas.configure(image=photo);state['last_image']=photo;state['drawn']=state['index']
                info=data.information(getattr(data,'displayed_index',state['index']));status.set(f"{'PAUSED' if state['paused'] else 'PLAYING'} {state['speed']}x | decision {info['global_decision']} | policy v{info['policy_version']}")
        except Exception as exc:status.set('Display error (training unaffected): '+repr(exc));record('display_error')
        root.after(35,tick)
    record('window_open');tick();root.mainloop()
    if data:data.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--group',choices=['R0_seed42','R0_seed123','R1_seed42','R1_seed123'],default='R1_seed42');p.add_argument('--episode',type=int,default=0)
    p.add_argument('--mode',choices=['actual','evaluation','comparison'],default='actual');p.add_argument('--checkpoint',type=Path);p.add_argument('--condition',choices=['nominal','gain90','mass110'],default='gain90')
    p.add_argument('--split',choices=['train','validation','test'],default='validation');p.add_argument('--live-stream',type=Path);p.add_argument('--speed',type=float,default=1.);p.add_argument('--paused',action='store_true');p.add_argument('--headless-export',action='store_true')
    p.add_argument('--decision',type=int)
    p.add_argument('--replay-only',action='store_true');p.add_argument('--other-run',type=Path,action='append',default=[])
    a=p.parse_args();session=destination('phase3d_viewer_'+str(os.getpid()))
    print('VIEWER SESSION',session,flush=True)
    if a.live_stream:gui(live=a.live_stream,session=session);return
    paths=[]
    if a.mode=='actual' or a.replay_only:
        if a.run is None:raise ValueError('--run is required; no implicit training or checkpoint replay')
        run=a.run/a.group if (a.run/a.group).exists() else a.run;paths=[run]+a.other_run
    else:
        if a.mode=='evaluation' and not a.checkpoint:raise ValueError('--checkpoint is required for deterministic evaluation')
        c={'name':a.condition,**({'kp_scale':.9} if a.condition=='gain90' else {'mass_inertia_scale':1.1} if a.condition=='mass110' else {})}
        jobs=[('Checkpoint',str(a.checkpoint))] if a.mode=='evaluation' else [('M3',None),('Integral',None)]+([('Checkpoint',str(a.checkpoint))] if a.checkpoint else [])
        for name,checkpoint in jobs:
            print(a.mode.upper(),'independent deterministic evaluation',name,flush=True)
            result=evaluate_job((sources(a.split)[0],c,name,checkpoint,str(session/name)))
            paths.append(Path(result['run_directory']))
        write_json(session/'independent_evaluation_results.json',dict(mode=a.mode,paths=paths,not_training=True))
    for path in paths:
        print('LOADING RECORDED EPISODE',path,a.episode,flush=True)
        data=RecordedEpisode(path,a.episode,a.mode)
        print('READY',len(data.t),data.fidelity,flush=True)
        if a.headless_export:
            import imageio.v2 as imageio
            video=session/f'{path.name}_episode{a.episode}_{a.mode}.mp4'
            with imageio.get_writer(str(video),fps=10,codec='libx264',macro_block_size=16) as writer:
                for i in range(0,len(data.t),50):writer.append_data(data.image(i))
            data.close();print('EXPORTED',video,flush=True)
        else:gui(data=data,session=session,speed=a.speed,paused=a.paused,decision=a.decision)
    write_json(session/'complete.json',dict(mode=a.mode,paths=paths));print('VIEWER OUTPUT',session,flush=True)


if __name__=='__main__':main_guard(main)

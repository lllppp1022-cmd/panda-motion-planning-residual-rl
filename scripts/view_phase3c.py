"""Serve only existing Phase3C TensorBoard events; never launches a training job."""
from _bootstrap import ROOT,main_guard
import argparse
from pathlib import Path
from scripts.phase3c_common import read
from src.rl.dependencies import activate
activate()


def main():
    p=argparse.ArgumentParser(description='查看已保存的Phase3C训练曲线，不启动新训练')
    p.add_argument('--port',type=int,default=6027);a=p.parse_args();spec=[]
    for stage in ('initial','continue'):
        pointer=ROOT/f'results/phase3c_training_{stage}_latest.json'
        if not pointer.exists():continue
        run=Path(read(pointer)['run'])
        for group in read(run/'complete.json')['results']:
            path=Path(group['run_directory'])/'tensorboard'
            spec.append(f"{group['version']}_seed{group['seed']}_{stage}:{path}")
    if not spec:raise RuntimeError('No completed Phase3C training event files')
    from tensorboard import program
    tb=program.TensorBoard();tb.configure(argv=[None,'--logdir_spec',','.join(spec),'--host','127.0.0.1','--port',str(a.port),'--reload_interval','10'])
    url=tb.launch();print('Existing actual training events:',url,flush=True)
    import threading
    threading.Event().wait()


if __name__=='__main__':main_guard(main)

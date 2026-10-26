from _bootstrap import *
import numpy as np
from src.robot.model_loader import prepare_model,load_model
from src.robot.panda_interface import PandaInterface
from src.robot.kinematics import Kinematics
from src.common import write_csv
def main():
    p=parser('Inspect actual Panda joints/servos and world-frame FK');p.add_argument('--q',type=float,nargs=7);a=p.parse_args();dest=destination('inspect_model')
    prepare_model();m,d=load_model();robot=PandaInterface(m,d);result=robot.inspection();write_json(dest/'model_inspection.json',result);print(__import__('json').dumps(result,indent=2))
    kin=Kinematics(m);qs=[kin.home,kin.home+np.array([.1,-.1,.08,-.12,.1,.1,.06])]
    if a.q:qs.append(np.array(a.q))
    rows=[]
    for index,q in enumerate(qs):
        x,R=kin.forward(q);row=dict(configuration=index,**{f'q{i+1}':float(v) for i,v in enumerate(q)},x_world_m=x[0],y_world_m=x[1],z_world_m=x[2],**{f'R{i}{j}':R[i,j] for i in range(3) for j in range(3)})
        rows.append(row);print('FK WORLD',q.tolist(),x.tolist(),'R=',R.tolist())
    write_csv(dest/'forward_kinematics.csv',rows);publish(dest,['model_inspection.json','forward_kinematics.csv'])
    write_json(dest/'complete.json',dict(passed=True,script='inspect_model',output=dest))
if __name__=='__main__':main_guard(main)

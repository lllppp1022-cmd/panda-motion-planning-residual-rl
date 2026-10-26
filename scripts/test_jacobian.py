from _bootstrap import *
import numpy as np
from src.robot.model_loader import load_model
from src.robot.kinematics import Kinematics
from src.common import config,write_csv
def main():
    parser('Central-difference validation of true3x7 arm Jacobian').parse_args();dest=destination('test_jacobian');m,d=load_model();kin=Kinematics(m);rng=np.random.default_rng(config()['sampling_seed']);rows=[];matrices={}
    for i in range(10):
        q=kin.home+rng.uniform(-.35,.35,7);engine=kin.compute_position_jacobian(q)
        assert engine.shape==(3,7)
        for eps in [1e-5,1e-6,1e-7]:
            fd=kin.finite_difference(q,eps);delta=engine-fd;rel=float(np.linalg.norm(delta)/max(np.linalg.norm(fd),1e-15))
            rows.append(dict(configuration=i,epsilon_rad=eps,rows=3,columns=7,max_abs_error=float(np.abs(delta).max()),relative_frobenius_error=rel,passed=rel<1e-2,q=q.tolist()))
            matrices[f'engine_{i}_{eps}']=engine;matrices[f'fd_{i}_{eps}']=fd
    write_csv(dest/'jacobian_validation.csv',rows);np.savez_compressed(dest/'jacobian_matrices.npz',**matrices);publish(dest,['jacobian_validation.csv'])
    result=dict(passed=all(r['passed'] for r in rows),configurations=10,comparisons=30,max_abs_error=max(r['max_abs_error'] for r in rows),max_relative_error=max(r['relative_frobenius_error'] for r in rows));write_json(dest/'complete.json',result);write_json(ROOT/'results/jacobian_summary.json',result);print(result);assert result['passed']
if __name__=='__main__':main_guard(main)

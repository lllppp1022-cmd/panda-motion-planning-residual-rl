from _bootstrap import *
import platform,sys,subprocess,importlib.metadata as metadata,json
import numpy as np
import mujoco
from src.robot.model_loader import load_model
def main():
    parser('Freeze independent environment and exercise real OpenGL rendering').parse_args();dest=destination('environment');m,d=load_model();renderer=mujoco.Renderer(m,height=240,width=320);renderer.update_scene(d);frame=renderer.render().copy()
    from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER,GL_VERSION
    def text(code):
        s=glGetString(code);return s.decode(errors='replace') if s else None
    gl=dict(vendor=text(GL_VENDOR),renderer=text(GL_RENDERER),version=text(GL_VERSION),rendered_shape=list(frame.shape),pixel_std=float(frame.std()),passed=frame.shape==(240,320,3) and frame.std()>1)
    from PIL import Image
    Image.fromarray(frame).save(dest/'render_check.png');renderer.close()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True)
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\Windows NT\CurrentVersion') as key:
            windows={name:winreg.QueryValueEx(key,name)[0] for name in ['ProductName','DisplayVersion','CurrentBuild','UBR']}
    except OSError as e:windows=dict(error=repr(e))
    packages=sorted((p.metadata['Name'],p.version) for p in metadata.distributions());versions=dict(packages)
    result=dict(python=sys.version,python_executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,independent_conda_environment=Path(sys.prefix).name=='panda_safe_rl',platform=platform.platform(),windows_registry=windows,gpu_query=gpu.stdout.strip(),gpu_query_returncode=gpu.returncode,gpu_claim_in_task='RTX4070',gpu_actual_differs_from_task='4070' not in gpu.stdout,mujoco=mujoco.__version__,numpy=np.__version__,opengl=gl,physics='CPU native mujoco.mj_step; NVIDIA/OpenGL is rendering, not GPU physics acceleration',installed_packages=versions,phase1_only_no_training_framework=not any(x.lower() in ['torch','stable-baselines3','tensorflow'] for x in versions))
    write_json(dest/'system_info.json',result);publish(dest,['system_info.json'])
    direct=['mujoco','numpy','scipy','matplotlib','PyYAML','pytest','imageio','imageio-ffmpeg']
    (ROOT/'requirements.txt').write_text('\n'.join(n+'=='+metadata.version(n) for n in direct)+'\n',encoding='utf-8')
    (ROOT/'requirements-lock.txt').write_text('\n'.join(n+'=='+v for n,v in packages)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='installed_packages'},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));assert result['independent_conda_environment'] and gl['passed'] and result['phase1_only_no_training_framework']
if __name__=='__main__':main_guard(main)

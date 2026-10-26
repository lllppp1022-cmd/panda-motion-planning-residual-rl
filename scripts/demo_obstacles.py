from _bootstrap import *
import numpy as np
from src.robot.model_loader import load_model
from src.robot.panda_interface import PandaInterface
from src.scene.obstacles import ObstacleScene
from src.robot.joint_controller import JointPositionController
from src.visualization.viewer import LiveViewer
def main():
    a=parser('Static geometry fixtures: clear, intentional external contact, moved away').parse_args();dest=destination('demo_obstacles');m,d=load_model();robot=PandaInterface(m,d);scene=ObstacleScene(robot);before=robot.positions();clear=scene.query()
    screenshots=[]
    def snapshot(name):
        if not a.record:return
        import mujoco
        from PIL import Image
        with mujoco.Renderer(m,height=480,width=640) as renderer:
            camera=mujoco.MjvCamera();camera.lookat[:]=[.2,.15,.45];camera.distance=2.6;camera.azimuth=135;camera.elevation=-25
            renderer.update_scene(d,camera=camera);path=dest/f'{name}.png';Image.fromarray(renderer.render().copy()).save(path);screenshots.append(str(path))
    snapshot('obstacles_clear')
    candidates=[g for g in scene.robot_ids if 'link4' in scene.name(g)];assert candidates
    center=d.geom_xpos[candidates[0]].copy();scene.move('obstacle_sphere',center);contact=scene.query();snapshot('obstacles_intentional_contact')
    scene.move('obstacle_sphere',[1.1,.8,.5]);moved=scene.query();snapshot('obstacles_moved_away');np.testing.assert_array_equal(robot.positions(),before)
    result=dict(passed=clear['minimum']['signed_distance_m']>0 and clear['external_contact_count']==0 and contact['minimum']['signed_distance_m']<0 and contact['external_contact_count']>0 and moved['minimum']['signed_distance_m']>0,clear=clear,intentional_collision=contact,moved_away=moved,fixture='Sphere placed at the world center of link4 collision mesh to create an explicit negative collision-query test. No colliding dynamic path executed.',robot_qpos_unchanged=True,api_signature=__import__('mujoco').mj_geomDistance.__doc__)
    # Viewer shows the normal clear scene; collided fixture is not dynamically executed.
    live=LiveViewer(robot,a.viewer,a.speed)
    if a.viewer:
        controller=JointPositionController(robot)
        try:
            for _ in range(1000):
                if not live.running():break
                with live.lock():controller.step(before)
                live.sync()
        finally:result['viewer']=live.close()
    else:result['viewer']=live.close()
    result['screenshots']=screenshots;write_json(dest/'obstacle_queries.json',result);write_json(dest/'complete.json',dict(passed=result['passed']));publish(dest,['obstacle_queries.json']);print('OBSTACLES',dict(passed=result['passed'],clear_min=clear['minimum']['signed_distance_m'],contact_min=contact['minimum']['signed_distance_m'],contacts=contact['external_contact_count'],moved_min=moved['minimum']['signed_distance_m'],robot_geoms=len(scene.robot_ids)));assert result['passed']
if __name__=='__main__':main_guard(main)

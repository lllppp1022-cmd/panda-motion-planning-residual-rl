"""Execute documented CLIs, retain logs and summarize actual pytest cases."""
from _bootstrap import *
import subprocess,json,xml.etree.ElementTree as ET
def main():
    dest=destination('acceptance_commands');commands=[]
    cases=[['scripts/check_environment.py'],['scripts/inspect_model.py'],['scripts/inspect_model.py','--q','0','0','0','-1.57079','0','1.57079','-0.7853'],['scripts/demo_joint_control.py'],['scripts/test_jacobian.py'],['scripts/demo_ik.py'],['scripts/demo_obstacles.py'],['scripts/demo_obstacles.py','--viewer','--record','--speed','3'],['-m','pytest','tests','-q','--junitxml='+str(dest/'pytest.xml')],['-m','pip','check']]
    for index,args in enumerate(cases):
        started=time.perf_counter();p=subprocess.run([sys.executable]+args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        (dest/f'command_{index:02d}.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
        commands.append(dict(arguments=args,exit_code=p.returncode,passed=p.returncode==0,elapsed_seconds=time.perf_counter()-started,log=str(dest/f'command_{index:02d}.log')));write_json(dest/'command_validation.json',commands);print('COMMAND',index,p.returncode,args,flush=True)
    tests=[]
    for case in ET.parse(dest/'pytest.xml').getroot().iter('testcase'):
        failure=case.find('failure');error=case.find('error');skipped=case.find('skipped');node=failure if failure is not None else error
        status='failed' if node is not None else 'skipped' if skipped is not None else 'passed';tests.append(dict(name=case.attrib['name'],status=status,seconds=float(case.attrib.get('time',0)),reason=node.attrib.get('message') if node is not None else skipped.attrib.get('message') if skipped is not None else None))
    result=dict(commands=commands,pytest_cases=tests,passed=all(c['passed'] for c in commands) and all(t['status']=='passed' for t in tests),counts={status:sum(t['status']==status for t in tests) for status in ['passed','failed','skipped']})
    write_json(dest/'command_and_test_validation.json',result);publish(dest,['command_and_test_validation.json']);assert result['passed'];print('ALL COMMANDS AND PYTEST PASSED',result['counts'])
if __name__=='__main__':main_guard(main)

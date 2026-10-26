"""Attach only actually completed regression results and immutable release evidence."""
from _bootstrap import ROOT,main_guard
from pathlib import Path
import xml.etree.ElementTree as ET
from src.common import sha,write_json
from scripts.phase3c_common import read
from scripts.train_phase3c import code_hashes


def main():
    suites=ET.Element('testsuites');test_count=0
    for name,expected in [('historical',173),('new',13)]:
        source=ROOT/f'results/phase3c_{name}_tests.xml';root=ET.parse(source).getroot()
        selected=[root] if root.tag=='testsuite' else list(root)
        count=sum(int(x.attrib.get('tests',0)) for x in selected)
        assert count==expected,(name,count)
        assert all(int(x.attrib.get('failures',0))==int(x.attrib.get('errors',0))==int(x.attrib.get('skipped',0))==0 for x in selected)
        for x in selected:suites.append(x)
        test_count+=count
    ET.ElementTree(suites).write(ROOT/'results/phase3c_tests.xml',encoding='utf-8',xml_declaration=True)
    log='\n\n'.join((ROOT/f'results/phase3c_{name}_tests.log').read_text(encoding='utf-8-sig') for name in ('historical','new'))
    (ROOT/'results/phase3c_tests.log').write_text(log,encoding='utf-8')
    prep=Path(read(ROOT/'results/phase3c_prepare_latest.json')['run']);history=read(prep/'history_sha256.json')
    assert all(sha(ROOT/name)==h for name,h in history.items())
    for stage in ('initial','continue'):
        training=Path(read(ROOT/f'results/phase3c_training_{stage}_latest.json')['run'])
        for group in read(training/'complete.json')['results']:assert group['source_hashes']==code_hashes()
    acceptance=read(ROOT/'results/phase3c_acceptance.json')
    assert acceptance['completed'] and len(acceptance['evaluation'])==42
    required=['docs/22_Phase3B训练诊断.md','docs/23_Phase3C奖励与训练方案.md','docs/24_Phase3C验收报告.md',
        'results/phase3c_training_comparison.csv','results/phase3c_evaluation_comparison.csv','results/phase3c_reward_value_audit.csv',
        'results/phase3c_action_effectiveness.csv','assets/plots/phase3c_learning_curves.png','assets/plots/phase3c_control_comparison.png']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size for p in required)
    from PIL import Image
    for path in required:
        if path.endswith('.png'):
            with Image.open(ROOT/path) as im:im.verify()
    acceptance['tests']=dict(passed=test_count,historical=173,new=13,skipped=0,failures=0,
        historical_files_verified=len(history),historical_integrity_passed=True,training_sources_unchanged=True)
    write_json(ROOT/'results/phase3c_acceptance.json',acceptance)
    doc=ROOT/'docs/24_Phase3C验收报告.md';text=doc.read_text(encoding='utf-8').split('\n## 实际工程验收记录')[0]
    text+=f'\n## 实际工程验收记录\n\n原173项+新增13项，合计{test_count}/{test_count}实际通过，0跳过、0失败。独立原始JUnit与终端日志合并到`results/phase3c_tests.xml`、`results/phase3c_tests.log`；未伪造单次测试时长。{len(history)}个历史文件SHA256均与开发前冻结清单一致；四组两阶段的训练源代码/配置哈希亦保持一致。两张PNG成功解码。\n\n'
    text+='本轮曲线已在`http://127.0.0.1:6027/`实际验证HTTP200、8个训练run及标量标签。6026仍可能展示旧3B日志；训练已完成，曲线停止追加是预期行为。查看当前已保存曲线（不启动训练）：\n\n```powershell\ncd D:\\mujoco_learning\\panda_safe_residual_rl\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/view_phase3c.py --port 6027\n```\n\n'
    text+='选择`train/value_loss`、`train/explained_variance`、`rollout/ep_rew_mean`；`recomputed_diagnostics`标签由实际已保存CSV/NPZ离线导出，含奖励分量和GAE分布，没有额外环境采样。R0到达奖励+200占主导，回报图变化小不等于Actor完全不变；同时查看真实TCP误差、实际残差及更新事件。\n'
    ev=acceptance['evaluation'];effect={};extra=0
    for v in ('R0','R1'):
        effect[v]={}
        for seed in (42,123):
            changes=[]
            for r in ev:
                if r['method']!=f'{v}_seed{seed}' or r['condition']=='nominal':continue
                m3=next(x for x in ev if x['method']=='M3' and x['condition']==r['condition'] and x['split']==r['split'])
                integ=next(x for x in ev if x['method']=='Integral' and x['condition']==r['condition'] and x['split']==r['split'])
                changes.append(100*(m3['tcp_rmse_mm']-r['tcp_rmse_mm'])/m3['tcp_rmse_mm'])
                extra+=int(r['tcp_rmse_mm']<integ['tcp_rmse_mm'])
            effect[v][str(seed)]=dict(mean_improvement_percent=sum(changes)/len(changes),min_improvement_percent=min(changes),max_improvement_percent=max(changes),all_four_improved=all(x>0 for x in changes))
    text+='\n## 六项研究结论\n\n'
    text+='1. **是否真正改善**：R0相对M3在本轮两个几何任务×两个失配工况×两个训练种子的8次配对中均降低TCP轨迹RMSE，但只有很小的数值收益。未证明存在显著或具有实际用途的控制优势。\n'
    text+='2. **是否跨两个种子**：R0满足本轮全部8次失配配对改善；R1未满足，seed123在部分条件下恶化。名义条件下两个奖励的seed123均比M3差，不能忽略这一取舍。\n'
    for v in ('R0','R1'):
        for seed in ('42','123'):
            e=effect[v][seed];text+=f"   - {v} seed{seed}：四个失配条件平均改善{e['mean_improvement_percent']:+.4f}%，范围[{e['min_improvement_percent']:+.4f}%, {e['max_improvement_percent']:+.4f}%]。\n"
    text+='3. **哪种奖励更合理**：R1的毫米误差尺度、较小终止奖励及EV更便于诊断价值学习，价值拟合明显改善；但真实控制结果未显示R1比R0稳定更好，不能仅凭较小loss选为获胜方法。两者定义均在最终测试前冻结。\n'
    text+=f"4. **安全代价**：四组训练以及42次正式验证/最终评价均无5mm违规和禁止接触，没有观察到安全退化；这仅限已经验证的宽裕度、轻度失配任务。\n5. **额外收益**：四个PPO在4个失配条件下超过受限积分补偿的次数为{extra}/16。积分在失配条件中明显更有效，但名义误差比M3大；保留全部结果，不宣称积分全工况最优。\n"
    recommendation='当前不建议直接扩大PPO预算。优先保留M3与受限积分作为强基线，只有确认仍存在传统方法无法有效解决、可重复且有实际幅度的残余问题后，再冻结新任务开展有限实验。' if extra==0 else '当前只支持进一步有界复核，不支持直接100k训练；先确认相对强传统补偿的收益可在新的冻结任务中复现。'
    text+='6. **是否扩大训练**：'+recommendation+'本轮已停止，不继续自动训练。\n'
    doc.write_text(text,encoding='utf-8')
    acceptance['research_conclusions']=dict(paired_improvement=effect,ppo_beats_integral_mismatch_cases=extra,total_mismatch_comparisons=16,recommendation=recommendation)
    write_json(ROOT/'results/phase3c_acceptance.json',acceptance)
    write_json(ROOT/'results/phase3c_release_manifest.json',dict(files={p:sha(ROOT/p) for p in required},
        tests=acceptance['tests'],checkpoints={g['checkpoint']:g['checkpoint_sha256'] for g in acceptance['training']},
        training_total_decisions=sum(g['decision_steps'] for g in acceptance['training']),training_total_physics_steps=sum(g['total_physics_steps'] for g in acceptance['training'])))
    print('FINAL ACCEPTANCE',acceptance['tests'],flush=True)


if __name__=='__main__':main_guard(main)

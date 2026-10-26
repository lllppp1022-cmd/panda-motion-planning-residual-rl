# GitHub公开发布验收报告

## 1. 公开发布基本信息

- 仓库：lllppp1022-cmd/panda-motion-planning-residual-rl
- URL：https://github.com/lllppp1022-cmd/panda-motion-planning-residual-rl
- 已核对认证账户：lllppp1022-cmd；默认分支：main。
- 实际公开：2026-10-26T15:50:46.945254+08:00（Asia/Shanghai）；官方API确认private=false、visibility=public。
- 本报告对应的已验收内容提交：`99f8f42424ed121cfb3702869aa7a928c903c6fb`。报告自身随后普通提交推送，不循环嵌入自身SHA；最终最新本地/远程HEAD另记本机runtime/public_anonymous_verification.json与runtime/public_final_git.json。
- 本机未安装gh；使用Git Credential Manager现有凭据和GitHub官方REST API执行可见性PATCH。凭据只在进程内存中使用，不写入脚本、日志或远程URL。

## 2. 公开前安全与许可审计

审计645个跟踪文件、全部可达引用的2次提交、609个独立blob（198144210字节），以及ZIP内493个条目中的文本元数据。远程引用核对只有main和其HEAD，未遗漏其他远程研究分支。检查Token/API密钥格式、私钥、嵌凭据URL、可疑硬编码密码、个人手机号/用户主目录及凭据文件名；结合源码、来源清单和机器人媒体检查，未发现阻止公开的项目。扫描不是对未知敏感信息的绝对保证。

公开范围沿用已审计的现有跟踪内容，新增仅为发布文档和验收证据。没有加入Git忽略的完整历史日志、其他checkpoint、原始CSV.gz或环境凭据。没有单文件超过25MiB。原始研究工程未写入。

Panda模型保留Menagerie原始Apache-2.0 LICENSE、README、CHANGELOG及来源commit cda5236f98383f6fbaa4820c6b18dba892cd087f；派生XML与声明保留。根目录第三方许可证目录不被冒用于自有代码。依赖由包管理器安装，未捆绑第三方库或FFmpeg二进制。自有代码、六模型与自制媒体尚未选定开源/再分发许可证；此次仅确认公开可见，不擅自授予MIT或Apache-2.0。

## 3. 公开资源检查

| 资源 | 结果 | 实际依据 |
|---|---|---|
| 中英文README | PASS | 匿名获取，内容SHA与本地相同，首页实际出现README表格 |
| 核心源码、Panda模型与许可证 | PASS | 远程树与本地全部跟踪路径一致，抽取核心文件匿名逐字节核对 |
| BC78 / BC85 / A42 / A123 / B42 / B123 | PASS | 六文件匿名下载、SHA匹配release/models.json；本轮六个模型加载并输出有限7维动作 |
| README预览GIF | PASS | 匿名获取、实际解码60帧、12.0秒，帧间存在变化 |
| project_overview.mp4 | PASS | 匿名获取、SHA一致、完整解码800帧/80秒 |
| phase3dc_actual_training.mp4 | PASS | 匿名获取、完整解码812帧/81.2秒，实际随机训练来源保留 |
| phase3dc_policy_comparison.mp4 | PASS | 匿名获取、完整解码2880帧/288秒，独立确定性评价来源保留 |
| 结果图、冻结指标、技术及面试文档 | PASS | 文件与主要链接匿名获取、SHA核对；48次冻结证据在隔离副本重算 |
| 浏览器内视频播放器人工逐项操作 | NOT VERIFIED | 本轮验证匿名HTTP资源及完整解码，没有把它说成GUI播放器人工验收 |

流程图为README内Mermaid；不是缺失的独立图片文件。第三方资源和媒体出处见THIRD_PARTY_NOTICES.md及release/video_edit_manifest.json。单条视频不是整体成功率证明。

## 4. 匿名访问证据

验证时间：2026-10-26T16:01:45.073396+08:00。使用独立HTTP opener，不传Authorization、不传Cookie、无Cookie处理器；不是用已认证API代替匿名检查。

- GitHub首页HTTP 200，可见项目名与TCP结果表；不是404或强制登录页面。
- 公共REST仓库接口HTTP 200，private=false、visibility=public、default_branch=main。
- 匿名main引用与本地HEAD一致；远程645个文件的路径集合完整匹配本地Git跟踪集合。
- README主要相对文件、图像、GIF、视频和模型通过raw.githubusercontent.com匿名GET获取并核对SHA，而非仅确认路径在树中。
- 模型共六个，已下载字节与本轮实际加载的本地模型SHA完全相同。

可移植证据见release/publication/；匿名网页响应与更完整下载记录留在被忽略的runtime/，不额外公开原始训练数据。

第一次匿名批量请求曾因TLS连接意外EOF中止；本机保留runtime/public_first_attempt_failure.txt。随后采用每请求最多3次的有界重试，全部所需资源通过。这个网络失败没有被解释为仓库不存在，也没有使用认证请求替代匿名验收。文档生成器还修正了一处Windows默认GBK读取UTF-8验收JSON的问题，未涉及任何实验代码或数据。

## 5. 实际轻量复现验收

解释器：D:/miniconda/envs/panda_safe_rl/python.exe。全部命令在本轮新建的隔离Git副本runtime/public_acceptance执行，避免重算入口写回发布目录的冻结CSV。

| 命令 | 结果 | 墙钟秒 |
|---|---|---:|
| `python scripts/setup_release.py --check` | PASS (exit 0) | 10.843 |
| `python scripts/portfolio.py doctor` | PASS (exit 0) | 2.529 |
| `python scripts/portfolio.py model` | PASS (exit 0) | 0.603 |
| `python scripts/portfolio.py plan` | PASS (exit 0) | 1.106 |
| `python scripts/portfolio.py policy` | PASS (exit 0) | 14.122 |
| `python scripts/portfolio.py training-replay` | PASS (exit 0) | 3.144 |
| `python scripts/portfolio.py evaluate --compact` | PASS (exit 0) | 2.827 |
| `python scripts/portfolio.py metrics --compact` | PASS (exit 0) | 1.748 |
| `python scripts/portfolio.py media` | PASS (exit 0) | 15.235 |

模型加载确认nq/nv/nu=9/9/8和274个碰撞配对；保存路径4个节点经0.005rad独立检查通过。六个checkpoint通过SHA、加载和有限7维预测。回放验证A_seed42训练现场RGB与状态，不重新predict/step生成替代轨迹。48个任务以float64证据重算，发布目录冻结CSV字节不变。没有新模型训练、超参数调整或完整48次动力学重跑。

另在发布目录执行 `python -m pytest -q --junitxml=runtime/public_release_tests.xml`：**8 passed in 3.61s**。这是本轮8项轻量测试，不能与历史222项或上次克隆测试相加冒充新增测试。九条入口命令也不算九项pytest。

## 6. Git远程一致性与文档变更

上面的内容基线本地/远程HEAD一致；最终文档普通提交后再检查匿名main、本地HEAD、git ls-remote一致及工作区干净。最终SHA记录在本机runtime/public_final_git.json，不把提交自身SHA循环写入本报告。

仅修改README.md、README_EN.md、LICENSE.md中的可见性与许可状态解释，并给32/33号历史报告追加后续状态；新增本34号报告和发布验收证据。算法、奖励、模型、原CSV与负结果不变。无强推、重写历史、删除分支或替代仓库。

## 7. 已知限制

- 未在全新电脑完整联网重装；当前学习依赖使用既有同版本只读回退路径，doctor如实披露。
- 没有真实Panda实机部署、Sim-to-Real验收或安全认证。
- 离散路径复核和仿真动态安全监控不构成连续时间安全证明。
- PPO未稳定超过Integral20；两个训练种子、有限几何与失配场景不支持统计显著或普遍优越宣称。
- 完整历史大数据未进入Git；保留本地归档与哈希索引，公开投影不能代替全部原始控制信号。
- 自有代码、模型和媒体尚无开源/再分发许可证；公众可访问不等于获得任意复用授权。第三方既有许可继续有效。
- 原生GUI、每键遥操作、网页播放器人工交互未在本次重新验收；保留既有证据并区分本轮范围。

## 8. 最终结论

**PUBLIC VERIFIED**（上述匿名验收对应的内容基线）。仓库已实际公开，匿名网页/API可访问，远程内容完整，六模型与主要README资源通过，轻量复现通过，未发现阻止公开的安全或许可问题。后续本报告提交的最新HEAD由外部最终日志闭环验证。

可以作为机器人算法实习作品集链接；简历应如实写明仿真研究、AI辅助工程过程、个人实际贡献、强积分基线以及PPO负结果，不虚构实机经验。

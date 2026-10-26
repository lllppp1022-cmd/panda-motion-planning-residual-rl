# GitHub发布准备报告

目标及实际已创建仓库：https://github.com/lllppp1022-cmd/panda-motion-planning-residual-rl

**当前可见性：Private。** 已验证登录账户为lllppp1022-cmd，并通过GitHub官方API创建/核对private=true；没有继承Reacher公开授权，没有执行visibility=public。

## 本地仓库

独立路径：`D:\mujoco_learning\panda-motion-planning-residual-rl`。原研究目录保持不变。main分支，首次提交前扫描敏感信息与大型文件，推送使用GCM认证，Token只在认证进程内存中处理，不写文件/日志/远程URL。

内容提交、远程HEAD及私有状态最终验收记录由release/remote_content_verification.json记录内容基线（若存在），最新HEAD可运行 `git rev-parse HEAD` 与 `git ls-remote origin refs/heads/main` 对照；不在同一提交内嵌其自身SHA造成循环。

## 许可与资源审计

- 自有源码、训练模型及媒体由用户和AI辅助工程形成；尚未选择向公众授予的开源许可证，因此LICENSE.md没有擅自授予MIT等权利。
- Panda目录Apache-2.0 LICENSE/README/CHANGELOG/资产原样保留，上游commit及派生XML修改说明见THIRD_PARTY_NOTICES.md。
- Menagerie根LICENSE为多模型许可证目录，不能把无关条目套到项目源码。没有复制第三方运行库、wheel或FFmpeg二进制。
- 大型原始训练日志、全部历史checkpoint和环境缓存保持本地。公开原始数据或额外大文件需另行检查，不因本次代码Private托管自动获得未来公开授权。
- 原始CSV用于本地复算；Git带紧凑float64证据及其来源SHA，清楚区分投影与完整控制日志。

## 推荐推送/验证方式

```powershell
Set-Location D:\mujoco_learning\panda-motion-planning-residual-rl
git status --short
git remote -v
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

首次设置origin并推送由收尾流程实际执行；如推送失败/未执行，以最终外部验收日志及本任务回复为准，不把只创建空仓库称为已上传。

## 公开前单独审核

需要用户明确确认本Panda仓库的Public权限、选用的自有源码开源许可证、模型/媒体的公开分发范围，以及简历个人职责表述。未确认前保持Private，不请求重复Reacher授权，也不自动公开。

## 最终发布补充验收

已实际推送内容提交 `fa13fb76d8f860101bf50e038316858c25e2e65c`，认证API读取远程main与本地一致；Private为true，README原始字节一致且GitHub Markdown渲染成功，远程640个文件的初始内容基线含必须的模型/媒体。此数字指首次内容基线，后续验收文档提交会增加文件，不应把该基线SHA称作未来最新HEAD。

最终文档提交后另做只读HEAD/Private复核，记录在本机runtime/remote_latest_verification.json，不把自身SHA循环写回提交。代码、模型、视频均未公开。公开权限、原始大数据分发范围、自有源码许可证及简历职责仍待本人审核。

# 底座与九种 LoRA 共存

网页选择“原版”时，生成、点赞和整句投喂沿用原来的底座流程；选择一种 LoRA 时，生成使用固定底座加对应适配器，标注和投喂只更新该适配器。规则抽卡的评价仍归入底座。

## 权重与训练归属

| 目标 | 权重入口 | 标注与持久队列 | 更新范围 |
| --- | --- | --- | --- |
| 原版底座 | `models/garbage-1/current.json` | `data/feedback.sqlite3` | 底座全部参数 |
| 九种角色 | `models/garbage-1/lora/<style>/current.json` | `data/lora/<style>.sqlite3` | 当前角色的 LoRA A/B |
| 角色依赖的固定底座 | `models/garbage-1/lora/base/current.json` | 不接收独立投喂 | 永不随在线训练更新 |

九种角色是嘴臭、疯子、娘炮、二次元、伪娘、社畜、兵王、外星人、陪酒女。它们共用同一版本的固定底座，但当前进程按需建立各自运行时；这不是显存共享优化，也不是混合专家。

首次制备将当时的底座 checkpoint 复制为固定底座。底座继续成长时，这份快照不会自动替换。每份适配器记录 `base_sha256`，不匹配则拒绝加载。网页“同步最新底座”显式启动重新适配与验证，成功后才切换角色版本。

每条生成记录保存 `model_run_id` 和 `model_step`。角色样本放在自己的 SQLite，反馈接口按 sample_id 找回目标，切换选择器不改变旧样本的归属。同一句话可以分别投喂给不同角色，重复判断只在各自数据库内生效。没有“同时训练底座和 LoRA”的默认行为。

## LoRA 本体

当前生产适配器为 FP32、rank 8、alpha 16，覆盖两层 Transformer 的 QKV、attention output、FFN up/down 投影。每种有 **47,952 个可训练参数**，纯参数数据 191,808 字节，未改动底座的 1,000,000 个参数。保留 rank 4 attention-only 格式兼容，首轮实验快照不会删除。

MLX 使用仅以 A/B 字典为可微分输入的梯度计算。PyTorch 兼容实现冻结底座并用 functional_call 注入低秩增量。推理暂时构造 W + (alpha/r)BA 后沿原计算路径执行；没有新增 KV cache，仍温度 0、全词表 argmax。CUDA 路径存在，但当前 Mac 仅实测 MLX 与 PyTorch CPU，未验证 CUDA 硬件。

在线训练与底座使用相同的持久任务模型。点赞/整句投喂执行一小步 SGD（LoRA 学习率 0.01），混入本角色训练集回放；点踩排除后续正向采样，不执行反向奖励或自动撤销历史学习。成功保存 adapter-only NPZ、元数据后原子更新指针；重启使用任务号恢复，防止重复更新。没有新增手动训练按钮。

## 初始训练与效果边界

`data/lora/styles.json` 各角色有 48 条训练句、8 条验证句，合计 504 条；没有把验证句放进训练回放或随机开场池。词表沿用底座的冻结词表，未知字走 UTF-8 字节回退。首轮 rank4 attention-only 跑了 240 步；扩大为 rank8 attention+FFN 后，每种跑 1,200 个候选训练步骤，以验证 loss 最低的检查点发布，因此发布步数可能小于 1,200。

初训用 Adam，在线用无动量 SGD。初训 Adam 状态不保存，训练脚本不是原优化器的断点续训。重训保留旧快照并产生新角色 run_id。

实测报告在 `output/lora-report.json`，包含验证 loss、固定输入在加载与关闭适配器时的真实输出，以及底座是否保持冻结。**这些适配器确实经过低秩训练，但语料很少，当前仍会出现重复、语法断裂和角色不鲜明。验证 loss 下降不能证明风格质量。** 显示文本没有用语料整句检索或规则生成替换。

制备命令（关闭其他正在使用相应角色的进程，避免 writer lease 冲突）：

```bash
.venv/bin/python scripts/train_lora.py --steps 1200
```

已有适配器默认跳过。明确重训某个角色、保留旧快照：

```bash
.venv/bin/python scripts/train_lora.py --styles mouthy --steps 1200 --retrain
```

网页只需选择角色、点 AI、标注和投喂。`GET /api/styles` 报告可用风格；`POST /api/ai`、`POST /api/feed` 支持 `style`，省略即为 `base`；`POST /api/feedback` 只需 sample_id 和 rating。

## 一键同步最新底座

风格选择器旁的“同步最新底座”对全部九种 LoRA 生效，不受当前选项影响。`POST /api/lora/sync` 提交空 JSON 即返回后台任务，`GET /api/lora/sync` 查询进度；网页每两秒查询，完成后刷新模型状态。重复提交正在执行的任务返回同一任务 ID。关闭网页不会取消工作，服务重启后恢复尚未完成的任务。

目标是点击时磁盘 `models/garbage-1/current.json` 指向的完整 checkpoint。同步期间原版仍能继续学习；新产生的底座版本留给下一次同步，避免目标在训练过程中变化。权重、配置和 tokenizer 元数据都校验。架构或 token ID 映射发生变化会明确拒绝迁移，保留原角色；此功能不包含词表扩容或网络结构转换。

目标底座写入不可变目录 `models/garbage-1/lora/bases/<权重摘要>-<配置词表指纹>/`。各角色 checkpoint 增加相对于 lora 目录的 `base_ref`，旧 checkpoint 没有该字段时仍读取 `lora/base`。每个角色可以独立切换，不会因为另一个角色失败而失去可用底座。

每个角色在自己的、实际持有跨进程写入锁的后台 worker 内执行迁移：先完成上一条普通训练任务，再暂停该角色的在线更新，用旧 LoRA A/B 在新冻结底座上进行 200 步 Adam 候选训练（学习率 0.0005），选验证 loss 最佳的候选。种子训练句排除已点踩文本，最多纳入 128 条人工正向句并加权；验证句不进入候选训练。选中的最佳候选可能就是训练前的重挂载结果，日志会如实记录 retained_step=0，不能把尝试步数当作最终保留步数。

验证门限是：新组合的验证交叉熵不得比原组合高超过 `max(0.05, 原 loss × 5%)`，同时要求 loss 有限、新底座摘要不变。通过后保存新的 adapter-only 快照、原子切换该角色的指针，并在短暂持有 runtime.lock 时热加载；失败继续使用旧版本。该数值门限不等于人类判断的角色风格或表达质量保证。

迁移计算期间生成使用旧权重；该角色的新投喂、评价继续写入原 SQLite 队列，待迁移结束后按顺序消费。原版和其他角色可照常处理。原版标注库不挪动，不会把 LoRA 合并进底座。任务及每角色结果保存在 `data/lora/sync-state.json`；已发布但任务状态尚未写完时，重启用 migration.job_id 和目标摘要识别，避免重复迁移。历史底座与角色文件均保留，没有新增自动清理或网页回滚按钮。

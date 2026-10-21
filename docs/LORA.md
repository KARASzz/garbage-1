# 底座与九种 LoRA 共存

网页选择“原版”时，生成、点赞和整句投喂沿用原来的底座流程；选择一种 LoRA 时，生成使用固定底座加对应适配器，标注和投喂只更新该适配器。规则抽卡的评价仍归入底座。

## 权重与训练归属

| 目标 | 权重入口 | 标注与持久队列 | 更新范围 |
| --- | --- | --- | --- |
| 原版底座 | `models/garbage-1/current.json` | `data/feedback.sqlite3` | 底座全部参数 |
| 九种角色 | `models/garbage-1/lora/<style>/current.json` | `data/lora/<style>.sqlite3` | 当前角色的 LoRA A/B |
| 角色依赖的固定底座 | `models/garbage-1/lora/base/current.json` | 不接收独立投喂 | 永不随在线训练更新 |

九种角色是嘴臭、疯子、娘炮、二次元、伪娘、社畜、兵王、外星人、陪酒女。它们共用同一版本的固定底座，但当前进程按需建立各自运行时；这不是显存共享优化，也不是混合专家。

首次制备将当时的底座 checkpoint 复制为固定底座。底座继续成长时，这份快照不会自动替换。每份适配器记录 `base_sha256`，不匹配则拒绝加载。以后迁移底座需要另做训练与验证，不能直接改指针。

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

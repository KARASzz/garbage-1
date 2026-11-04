# 嘴硬终端 · 升级方案与 10M 底座演进实施计划 (Implementation Plan)

> **目标**：分步完成当前 1M 项目功能升级（采样解码多样性、自定义 Prompt 接梗、因果常识注入管线），随后将模型架构升级至 1000 万参数（10M），并完成 10M 底座预训练，最终交接给用户进行 9 种 LoRA 自主投喂观察。

---

## 阶段规划概览

```text
阶段一：升级现有 1M 项目 (Quick Wins & Polish)
  ├─ 1.1 解码策略升级：MLX / PyTorch 支持 Temperature / Top-p / Repetition Penalty
  ├─ 1.2 交互升维：打通自定义 Prompt / 挑衅输入框，支持直接指定开头让 AI 回怼
  └─ 1.3 因果注入：制备多角度因果语料 data/causal_commonsense.jsonl + 编写底座注入脚本

阶段二：模型架构升级到一千万参数 (10M Architecture)
  ├─ 2.1 架构泛化：支持任意 N 层 Transformer（原先硬编码 2 层）
  ├─ 2.2 10M 黄金配方设计：层数 6~8，d_model ~384，精确 10,000,000 FP32 参数解算
  └─ 2.3 适配器升级：LoRA 模块支持多层 Transformer 与 Rank 8/16 动态初始化

阶段三：完成 10M 底座预训练与环境就绪 (10M Base Pre-training & Handoff)
  ├─ 3.1 语料池扩增：混合语法槽位抽卡、因果常识、修辞反差与神评对话语料
  ├─ 3.2 执行 10M 底座预训练：基于 MLX / Metal 进行全量迭代，监控收敛与留出 Loss
  └─ 3.3 交付就绪：重置 9 个 LoRA 初始槽位，挂载到 10M 新底座，供用户在 Web 端自主投喂
```

---

## 详细任务与文件清单

### 任务 1：升级解码策略 (Temperature / Top-p / Repetition Penalty)
**涉及文件：**
- `trash_talk/mlx_backend.py`
- `trash_talk/torch_backend.py`
- `trash_talk/lora.py`
- `tests/test_sampling.py`

- [ ] **Step 1.1**：在 `MLXBackend.generate` 和 `MLXLoRA.generate` 中实现核采样（Top-p）、温度调节（Temperature）、以及重复惩罚（Repetition Penalty），当 `temperature=0` 时无缝回退至确定性 Argmax。
- [ ] **Step 1.2**：在 `TorchBackend` / `TorchLoRA` 中同步实现相同的采样逻辑。
- [ ] **Step 1.3**：编写单元测试验证采样稳定性（避免 NaN/Inf，验证重复惩罚有效性）。

### 任务 2：打通自定义 Prompt / 接梗回怼通道
**涉及文件：**
- `trash_talk/server.py`
- `web/index.html`
- `web/app.js`
- `web/style.css`

- [ ] **Step 2.1**：在 `POST /api/ai` 中增加可选参数 `prefix`（自定义起手句）与 `temperature`。
- [ ] **Step 2.2**：在 Web 界面增加挑衅/接梗输入框（支持空置抽卡，或输入指定开头如“人吃了毒药”）。
- [ ] **Step 2.3**：测试端到端输入，确认输入自定义开头后 AI 能够依循因果接下去。

### 任务 3：构建“因果常识与人间真实”注入管线
**涉及文件：**
- `data/causal_commonsense.jsonl`
- `scripts/train_causal_base.py`

- [ ] **Step 3.1**：按“多角度强化法”编写高密度因果与生存/人性常识语料（毒药与死亡、火与烫伤、穷富与虚荣、虽然/但是转折句）。
- [ ] **Step 3.2**：实现 `train_causal_base.py`，配置 shuffle、mini-batch (8~16)、学习率 3e-4、Loss 降至 1.5~1.8 时熔断早停以防复读。
- [ ] **Step 3.3**：运行脚本对当前 1M 底座进行注入，并测试一键 LoRA 同步。

### 任务 4：10M 模型架构设计与泛化重构
**涉及文件：**
- `trash_talk/garbage_model.py`
- `trash_talk/mlx_backend.py`
- `trash_talk/torch_backend.py`
- `trash_talk/lora.py`

- [ ] **Step 4.1**：重构 `garbage_model.py`，使层数支持动态 N 层（解绑固定的 2 层限制）。
- [ ] **Step 4.2**：实现 `solve_config` 针对 10,000,000 参数的精确求解（层数 6~8，d_model 384）。
- [ ] **Step 4.3**：让 `MLXBackend` 和 `TorchBackend` 动态根据配置循环前向传播。
- [ ] **Step 4.4**：更新单元测试，验证 10M 架构在 PyTorch 与 MLX 下的前向一致性（Parity Check）。

### 任务 5：10M 底座语料准备与预训练执行
**涉及文件：**
- `scripts/prepare_10m_data.py`
- `scripts/train_garbage.py`
- `models/garbage-10m/`

- [ ] **Step 5.1**：扩增 10M 预训练语料（结合规则语法槽位全组合、因果世界模型、民间谚语与歇后语、幽默对话）。
- [ ] **Step 5.2**：使用 MLX 在 Mac Metal 上执行 10M 底座预训练，监控 Loss 下降曲线。
- [ ] **Step 5.3**：保存原子检查点 `models/garbage-10m/current.json`。
- [ ] **Step 5.4**：初始化 9 个 LoRA 初始适配器槽位，更新前端风格与状态配置。
- [ ] **Step 5.5**：整体验收，交接给用户通过 Web 端“投喂”和“点赞点踩”观察各自 LoRA 的生长变化。

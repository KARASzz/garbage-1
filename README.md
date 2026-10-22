```text
   _________    ____  ____  ___   __________     ___
  / ____/   |  / __ \/ __ )/   | / ____/ ____/    <  /
 / / __/ /| | / /_/ / __  / /| |/ / __/ __/______/ /
/ /_/ / ___ |/ _, _/ /_/ / ___ / /_/ / /__/_____/ /
\____/_/  |_/_/ |_/_____/_/  |_\____/_____/     /_/

        LOCAL CRT TERMINAL // 1M PARAMS // FP32
```

<div align="center">

# 嘴硬终端 · garbage-1

**本地开机，按下开喷。把你的评价，写进它的权重。**

`LOCAL FIRST` · `MLX / METAL` · `1,000,000 PARAMS` · `FP32` · `9 × LoRA`

[快速启动](#boot) · [交互与学习](#learn) · [模型规格](#model) · [角色适配器](#lora) · [开发手册](#dev) · [HTTP API](#api)

</div>

---

一个跑在本机的 CRT 垃圾话终端。**规则抽卡**从 SQLite 按中文语法槽位拼句；**手搓 AI**运行自己从零训练的 `garbage-1` 字符语言模型。生成、点赞、点踩、投喂和改写都在本地完成，后台负责更新权重。

无需 API 密钥或在线模型服务。默认使用 Apple Silicon Mac 的 **MLX / Metal**，首次安装依赖后可断网运行。

| 通道 | 引擎 | 输出方式 |
| :--- | :--- | :--- |
| `RULE` · 规则抽卡 | SQLite + 中文语法槽位 | 按类别、搭配标签与句型抽取语料块 |
| `AI` · 手搓 AI | garbage-1 · 1M · FP32 | 随机短开场 + 逐 token 全词表 argmax，温度 0 |

![嘴硬终端的 CRT 页面与生成按钮](output/playwright/garbage-1-desktop.png)

<details>
<summary><b>ARTWORK / 查看项目海报</b></summary>

![嘴硬终端 · garbage-1 海报](poster-garbage-1.png)

</details>

<a id="boot"></a>

## 00 / boot · 开机

**双击 [启动.command](启动.command)。** 浏览器会自动打开，保留终端窗口；在终端按 **Control+C** 关闭服务。再次双击会复用本项目同版本、同推理后端的服务。

```bash
# 或者，在项目目录手动启动
./setup.sh
.venv/bin/python run.py --open
```

| 启动项 | 要求 / 行为 |
| :--- | :--- |
| 系统 | Apple Silicon macOS |
| Python | 3.12 或更新版本 |
| 本机地址 | <http://127.0.0.1:8765> |
| 端口占用 | 自动尝试后续可用端口，以终端显示的地址为准 |
| 安装位置 | 仅创建项目内的 `.venv` |
| 固定依赖 | APSW `3.54.1.0` / SQLite `3.54.1` · MLX `0.32.4` · NumPy `2.5.4` · PyTorch `2.14.1` |

复制到另一台机器时重新创建 `.venv`，**保留 `data/` 和 `models/`**：语料、人工评价、训练队列和模型权重都在这里。

<a id="learn"></a>

## 01 / learn · 教它说人话

日常操作都在网页里完成，投喂和评价会自动入队，**无需训练按钮**。

| 你的操作 | 后台发生什么 |
| :--- | :--- |
| 点击任意生成按钮 | 保存待评价样本，记录来源、主题、模型 run 与步数；未评价的句子不会自动当作好句子 |
| 点赞 | 加入正向样本池，与此前点赞句、可信训练种子混合，执行一个 FP32 SGD 步骤 |
| 点踩 | 排除后续正向训练与种子回放中的同文句子 |
| 在显示器下方投喂整句 | 保存为正向语料并排入监督训练队列；不拆词，相同整句只收录一次 |
| 点击屏幕句子，编辑后提交改写 | 保留原句作为 rejected，改写作为 chosen，执行 DPO + 监督训练 |
| 点击“同步最新底座” | 后台迁移全部九种 LoRA，显示独立同步进度 |

```text
 like / feed ──> supervised queue ──┐
                                   ├──> MLX worker ──> checkpoint
 rewrite ──────> preference queue ──┘                        │
                                                            v
 dislike ──────> exclude from positive sampling       atomic pointer
```

SQLite 在同一事务中保存评价与后台任务。请求无需等待梯度计算，生成按钮可以继续使用；**只有权重实际改变并成功保存 checkpoint，训练步数才会增加**。

同文句子的评价去重，重复点同一评价不重复建任务，评价可以反转。重启会恢复未完成任务，checkpoint 中的任务编号避免“权重已保存、任务尚未确认”时重复更新。人工评价和投喂句使用独立数据库，重新导入规则语料不会删除它们。

### `rewrite` / 直接改写

生成后点击显示器里的句子，直接编辑，再点 **“提交改写”**。“恢复原文”可以取消当前编辑；点赞、点踩仍评价原来的生成句。

选原版训练底座，选角色训练该 LoRA，**按生成时的归属处理**。下方整句投喂照常使用。保存提示表示入队，实际梯度计算在后台进行；旧页面已有结果可能缺少生成上下文，请刷新后重新生成。

算法、数据与限制见 [偏好训练说明 →](docs/PREFERENCE.md)。

> **学习边界**：点赞与整句投喂属于人工筛选后的增量监督学习。点踩只提供排除信息，不会自动撤销过去的梯度，也不表示模型知道该怎样改写；单独点踩不会执行 DPO。每次更新只是一小步，不保证立刻改变下一句。

<a id="model"></a>

## 02 / inspect · 一百万参数，逐项可查

| 模型规格 | 当前值 |
| :--- | :--- |
| 架构 | 两层 pre-LN causal Transformer · dense |
| 唯一可学习参数 | **1,000,000**，全部有效参数，无占位权重 |
| 精度 | 全部 **FP32** |
| 隐藏宽度 / 注意力头 | `160` / `8` |
| Context | `174` token |
| FFN 宽度 | 第 1 层 `858` · 第 2 层 `859` |
| 词表 | `1,323` = 1,059 个训练字符 + 8 个控制 token + 256 个 UTF-8 字节回退 token |
| 权重绑定 | 输入 embedding 与输出投影共享权重 |
| 裸参数体积 | `4,000,000` 字节，NPZ 文件另有格式开销 |
| 解码 | 全词表 argmax · 温度 `0` |
| 网页开场 | 从训练集形成的短开场池随机选择 **3～6 个字** |

相同权重、相同开场，输出固定。每次点击的多样性来自输入开场与后续权重更新；开场之后全部由模型逐 token 续写，AI 结果不会回退成规则抽卡。

它仍是小型娱乐实验，可能重复、语法不通或不够狠。规则按钮继续提供按句型组织的输出。

### `corpus` / 语料库存

| 规则库 | 规模 |
| :--- | :--- |
| 分类 / 句型 | `14` 类 / `40` 个句型 |
| 语料块 | **1,173** 条 = 1,100 条抽取池 + 73 条储备 |
| 扩充记录 | 旧的 917 条记录保留，新增 256 块 |
| 单次抽取 | 按类别、搭配标签与可选槽位抽取 `3～5` 块 |

维护规则语料见 [data/CORPUS.md →](data/CORPUS.md)。

<details>
<summary><b>DATA / 初训数据、字符编码与能力边界</b></summary>

初训数据由实际规则生成器扩增 16,000 句，再加 120 句人工整理的种子句。完整句去重后，固定划分为：

| 划分 | 句数 |
| :--- | ---: |
| Train | 14,508 |
| Validation | 806 |
| Test | 806 |

词表只在训练集拟合。种子数据明确标记来源，**不是用户点赞数据**。初训使用本机 MPS 跑了两轮，覆盖完整 context；旧 100K checkpoint 保留在版本化文件中。

英文字母 A–Z、a–z 共 52 个均已显式登记，各占一个 token、大小写分离，复用现有 ASCII 字节编号，因此无需扩容或迁移旧权重。词表外汉字通过 UTF-8 字节回退无损编码，可以进入投喂训练；可表示一个字，不等于自动拥有这个字或话题的语言知识。

训练、验证、测试共享底层模板与语料块。留出指标只说明域内拟合，不能据此声称通用中文能力或未见模板泛化。

模型、数据指纹、训练指标与 1M 升级记录见 [模型说明](models/garbage-1/MODEL_CARD.md) 和 [升级记录](output/garbage-1m-upgrade.json)。

</details>

<details>
<summary><b>CONTENT / 内容处理约定</b></summary>

应用不对**黑话、脏话或成人话题**设置关键词审查、拒答模板、星号替换或生成后的内容过滤。原文按普通文本保存、显示并进入标注流程；语法标签只决定规则抽卡时的搭配，你主动点踩后的排除属于个人偏好学习。训练与推理链路没有额外的内容审核服务或拒答训练阶段。

模型能表达的内容仍取决于训练语料。UTF-8 字节回退保证可表示，不会自动带来相关语言知识。

</details>

<a id="lora"></a>

## 03 / adapters · 九种 LoRA，共存生长

```text
 base                  -> update base weights
 frozen base + style   -> update selected LoRA only

 [01] mouthy     [02] mad        [03] sassy
 [04] anime      [05] femboy     [06] office
 [07] soldier    [08] alien      [09] hostess
```

网页风格选择器支持**原版 + 九种独立角色**：嘴臭、疯子、娘炮、二次元、伪娘、社畜、兵王、外星人、陪酒女。

选原版时继续训练底座；选角色时只训练该角色的 LoRA。角色使用单独固定的底座快照，原版后续更新不会自动改变角色依赖。各自的标注库、训练队列、版本指针相互隔离，点赞按生成样本归属处理。

需要迁移时，点击 **“同步最新底座”**，后台处理全部九种 LoRA。具体目录、训练方法、CUDA 兼容与当前效果边界见 [LoRA 说明](docs/LORA.md)，实际训练指标和输出对比见 [训练报告](output/lora-report.json)。

<a id="backends"></a>

## 04 / backends · MLX、CUDA 与扩展接口

推理后端通过 registry 接入。MLX 和 PyTorch 共享唯一参数名、FP32 NPZ 权重与 tokenizer；预留的 CUDA 适配器使用同一模型结构与贪心逻辑。

```bash
# 在具有 CUDA 的主机上，安装对应 PyTorch 后运行
.venv/bin/python run.py --backend cuda --port 8865
```

| 后端 / 架构 | 验证与实现边界 |
| :--- | :--- |
| MLX / Metal | 默认本机后端 |
| PyTorch CPU | 当前 Mac 已验证与 MLX 的权重兼容 |
| CUDA | 适配器已提供；设备不可用时明确报告模型未就绪；**尚未在 CUDA 硬件实测** |
| MoE | 当前只有 dense 模型，保留扩展接口，尚无路由器与专家 |

<details>
<summary><b>ARCH / 后续 MoE 接入约定</b></summary>

FFN / architecture registry、FFN 辅助损失接口、checkpoint 格式版本和 `run_id` 为后续 MoE 留出接口。升级时需要实现路由器、专家和负载均衡损失，并另建模型 run；未知架构会拒绝加载，保留原权重。

标注库独立于架构，可以继续复用。未来 MoE 应分别统计总参数与每 token 激活参数，不能沿用本版密集模型的 1M 声明。

</details>

<a id="files"></a>

## 05 / filesystem · 数据落在哪里

| 路径 | 职责 |
| :--- | :--- |
| `data/corpus.json` / `data/trash_talk.sqlite3` | 可编辑规则源 / 启动时事务导入的 SQLite |
| `data/feedback.sqlite3` | 人工样本、评价事件、持久后台任务、训练结果 |
| `data/garbage-1/` | 带来源的训练句、固定划分、指纹、短开场与回放种子 |
| `models/garbage-1/current.json` | 当前 checkpoint 的原子指针 |
| `models/garbage-1/checkpoints/` | 初训与在线更新的不可变权重和元数据 |
| `trash_talk/garbage_model.py` | 字符 tokenizer、精确参数预算、参考网络及 MoE 扩展接口 |
| `trash_talk/mlx_backend.py` / `trash_talk/torch_backend.py` | 本机 MLX / CUDA 与 CPU 兼容适配器 |
| `trash_talk/model_runtime.py` / `trash_talk/feedback.py` | 模型加载、静默 worker、事务标注和任务恢复 |
| `web/` | 纯本地网页资源 |

<a id="dev"></a>

## 06 / dev · 开发与检查

修改规则语料后重启，即可重新导入规则库；模型的 tokenizer 映射和权重不会自行重建。当前 1M 词表带 UTF-8 字节回退，投喂新汉字无需改词表。

### `train` / 重新初训

如需重新制备初训数据并开启新的模型 run，**先关闭服务**，再运行：

```bash
.venv/bin/python scripts/prepare_garbage.py
.venv/bin/python scripts/train_garbage.py --target-parameters 1000000 --epochs 2 --device mps
```

该命令从零初训，保留已有 checkpoint 文件并切换当前指针；不会继续原 AdamW 优化器。日常使用不需要运行它。在线训练使用无动量 SGD，权重、步数和已处理任务号足以恢复，不依赖未保存的优化器状态。

### `check` / 本地验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit.py --count 10000
```

<a id="api"></a>

## 07 / api · 本机 HTTP 接口

默认服务地址：`http://127.0.0.1:8765`，实际端口以终端输出为准。

| 方法 | 路径 | 用途 |
| :--- | :--- | :--- |
| `GET` | `/api/health` | 实际 SQLite 版本、语料源指纹与推理后端 |
| `GET` | `/api/model` | 模型 / 队列诊断；网页不展示训练状态 |
| `GET` | `/api/styles` | 可用角色风格 |
| `POST` | `/api/draw` | 规则生成，登记可评价样本 |
| `POST` | `/api/ai` | 模型生成，登记可评价样本 |
| `POST` | `/api/feedback` | 保存评价 |
| `POST` | `/api/feed` | 接收一条完整中文句，自动排队训练 |
| `POST` | `/api/correct` | 提交生成句的改写，排入偏好训练队列 |
| `POST` | `/api/lora/sync` | 提交全部九种 LoRA 的底座同步任务 |
| `GET` | `/api/lora/sync` | 查询同步进度 |
| `GET` | `/api/draw?debug=1` | 纯规则调试，不登记标注样本 |

---

`END OF MANUAL` · `KEEP A STRAIGHT FACE` · [↑ 回到开机](#boot)

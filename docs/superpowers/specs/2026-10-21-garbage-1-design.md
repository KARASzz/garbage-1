# garbage-1：标注驱动的静默训练

最新需求覆盖先前“点击生成时同步训练”的设计：不提供训练按钮；保存点赞/点踩后，后台持久队列静默处理，生成请求不等待梯度步骤。本机推理和线上更新使用 MLX/Metal，保留 CUDA 适配和未来 MoE 开发接口。当前精确 100000 个有效参数、FP32、温度 0 全词表 argmax。

两个生成入口（规则抽卡、手搓 AI）共用 CRT，输出后出现赞/踩。规则结果和 AI 结果保存为 pending，不自动当作正样本。评分变更事务性写事件和队列；同文最终评分去重、重复评分幂等。后台用最新点赞句与可信训练种子回放做小步 SGD；点踩排除未来正向训练和回放，不宣称撤销历史梯度或完整 RLHF。

原始模型为手写两层 causal pre-LN Transformer，字符 tokenizer、主题 token、tied embedding/output 和完整 context 位置表。实际词表1067时，d=48、heads=6、context=192、FFN=[98,99]，恰好100000个唯一参数。PyTorch用于初训和CPU/CUDA参考，MLX功能实现读取同名canonical FP32数组。不能在AI路径用规则结果回退伪装模型。

初始数据：16000条实际规则生成句+120条curated_seed，稳定完整句hash划分train14508/validation806/test806。tokenizer只在train拟合。底层模板/词块跨划分共享，指标只说明域内拟合；curated种子不得伪装成人类赞。

评价数据独立在data/feedback.sqlite3，保存样本来源、文本摘要、评分事件、run_id、模型步数、后台任务和训练结果。重导入规则语料不能触碰此库。后台原子领取任务，启动时恢复running任务；权重checkpoint保存last_job_id避免保存权重后崩溃导致重复梯度。版本化NPZ和元数据完成后才原子替换current.json；失败恢复内存权重，不计成功步骤。

同checkpoint、同prompt贪心结果应一致。随机性来自输入的3～6字开场和之后发生的模型更新，推理本身无采样、top-k或repetition mask。页面不显示后台训练步数/进度。

MoE边界：当前仅dense；architecture/FFN registry及辅助损失接口、checkpoint格式版本、run_id为扩展点。未来实现路由/专家/负载均衡及显式权重迁移，新建run并分别报告总参数/激活参数。未知架构拒绝加载，不重置原模型。独立人工标注可继续复用。

CUDA边界：canonical权重+独立Torch适配器，要求显式可用CUDA设备，无设备明确失败；本机Mac只能验证CPU映射和MLX parity，不能声称CUDA硬件实测。

验收包含参数/FP32/因果性、跨后端parity、纯argmax确定性、真实权重改变与重启恢复、评价队列幂等/崩溃边界、Origin/JSON输入检查、模型缺失时规则可用、浏览器四按钮与移动布局、实际macOS双击启动。

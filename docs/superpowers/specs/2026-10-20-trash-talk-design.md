# 垃圾话生成器设计

用户已批准实施：本地网页，仅一个 CRT 显示器和一个按钮；允许脏字；不使用 LLM，按中文语法类别从 SQLite 语料块抽取并组合；持续实施至可运行、可验证。

采用 Python 标准库 HTTP 服务、项目独立 SQLite 引擎、原生网页。监听 127.0.0.1。SQLite 目标版本以实施时官方稳定版为准，启动查询 sqlite_version() 证明实际引擎。数据库和依赖保存在项目内，首次安装后离线使用。

语料源 data/corpus.json：顶层 version、categories、fragments、templates。categories 记录 id/label/description；fragments 记录 id/category/text/tags/weight；templates 记录 id/name/theme/weight/segments。segments 为字符串或槽位字典，字典包含 slot/category，可带 tags_all/tags_any/exclude_tags、optional（0 到 1 的出现概率）、prefix/suffix。文本不得含未解析占位符。固定结构由句型控制，可选成分由槽位控制；限定类别和标签同时保证词性、配价、语义搭配。每个句型至少三个必选语料槽，允许随机可选槽；关联词、数量词和标点由句型或完整短语配套提供。

SQLite 规范化保存 categories、fragments、tags、fragment_tags、templates、template_slots 和 compatibility_rules，使用 STRICT、外键、参数化 SQL。模板每段结构与查询约束均可审计。运行时加载已验证的类型池，按权重选模板和槽位语料；有限重试，检测重复和长度；保留近期句子/块去重。API 返回 text、id、theme、block_count；调试信息仅明确 debug 请求返回。health 返回 app、sqlite_version、corpus counts。

前端 GET /api/draw 获取新句子；GET /api/health 只作启动和验证。所有字体和资源本地保存，CSS 构成 CRT 机壳、扫描线、辉光、背景。文本通过 textContent 显示；键盘可操作、减少动态偏好、移动屏幕可完整阅读；忙碌状态禁用按钮，失败可同按钮重试。无额外设置或语料管理 UI。

交付包含数据源、生成后的 SQLite 数据库、启动.command、安装/运行程序、说明及实际验证产物。验收包括数据库完整性、外键、语料引用、每种模板覆盖、批量生成多样性和规则约束、真实 HTTP API、桌面/移动网页点击和长句显示、错误恢复、离线资源检查。

# 垃圾话生成器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 交付可双击启动、离线抽取中文嘴臭句子的单按钮 CRT 网页。

**Architecture:** Python 提供同源静态资源与 draw/health API，SQLite 保存分类型短语和句型，生成器按类别和标签限定池加权抽样。独立语料和页面任务只修改各自文件，集成及复核由主线程执行。

**Tech Stack:** Python 3、SQLite 官方当前稳定版、APSW 或等效独立 SQLite 适配、HTML/CSS/JavaScript。

---

### Task 1: 明确接口和数据契约

- [x] 设计保存至 docs/superpowers/specs/2026-10-20-trash-talk-design.md。
- [x] 数据契约：categories/fragments/templates；槽位支持 category/tags_all/tags_any/exclude_tags/optional/prefix/suffix；文本不可含占位符。
- [x] API 契约：GET /api/draw 返回 {text,id,theme,block_count}；GET /api/health 返回运行版本和数据库规模。

### Task 2: 语料与句型

**Files:** data/corpus.json、data/CORPUS.md。

- [x] 制作至少 500 条实际短语记录、至少 24 种句型和不少于 12 个语法类别；所有类别/标签引用有效。
- [x] 每种句型至少三个必选槽，并包含可选槽；按照主题固定搭配，避免同一语料池相互不兼容。
- [x] 执行 `python3 -m json.tool data/corpus.json > /dev/null`；检查 ID 唯一、非空文本、正权重、槽位有候选。保存语料维护说明。

### Task 3: CRT 页面

**Files:** web/index.html、web/style.css、web/app.js、web/favicon.svg。

- [x] 制作一个显像管屏幕与一个语义 button，GET /api/draw 后通过 textContent 显示句子。
- [x] 完成机壳、扫描线、辉光、开机和短促文字动画；响应移动宽度及 prefers-reduced-motion。
- [x] 错误/等待/键盘状态与中文长句换行，字体资源不请求外部网络。
- [x] 执行 `node --check web/app.js`，由主线程浏览器检查布局和点击。

### Task 4: 数据库、引擎及 HTTP 服务

**Files:** trash_talk/storage.py、trash_talk/generator.py、trash_talk/server.py、run.py、schema.sql、tests/test_generator.py。

- [x] 安装项目隔离依赖，查询实际 `SELECT sqlite_version()`，验证达到当前稳定版本。
- [x] 创建 STRICT 表和外键，用事务导入 JSON；验证完整性和语料可生成性，已有数据库按源摘要判断更新。
- [x] 实现加权、可选槽、固定标签过滤、近期去重和有限重试；支持固定随机种子及调试抽样记录以供测试。
- [x] 提供仅本机地址的静态页面、draw、health 路由，拒绝路径穿越，错误返回 JSON。
- [x] 执行 `.venv/bin/python -m unittest discover -s tests -v`；批量生成 10000 条，覆盖所有句型、检查规则/长度/去重。

### Task 5: 启动、集成与复核

**Files:** 启动.command、setup.sh、README.md、.gitignore、output/。

- [x] 双击文件自动检查环境、启动或复用本项目服务、打开浏览器；首次安装失败输出可操作错误。
- [x] 实际启动后通过 health 校验引擎、数据库和 API；浏览器检查桌面/移动/按钮/长句/错误恢复/无外部请求。
- [x] 需求符合性和代码质量分阶段审查；修复发现的问题后仅重跑受影响验证。
- [x] 记录实际语料规模、SQLite 版本和验证证据，交付启动入口及本地 URL。

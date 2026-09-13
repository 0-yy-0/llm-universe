# C7 README Information Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 C7 教程入口、章节归属和评估任务中的信息架构问题，让读者能够从先修指标开始，按 RAG 执行阶段选择方法，并忠实记录实验结果。

**Architecture:** 保留现有七章主线和 Notebook 文件路径，只调整 README 导航、跨章链接和少量 Notebook 标题。教程首页负责学习路径与最小指标定义；章节 README 负责方法选择与边界；实现、审核和维护细节继续由 Notebook 与维护说明承载。

**Tech Stack:** Markdown, Jupyter Notebook metadata, Python validation scripts

---

## Task 1: 补齐指标先修入口并消除编号歧义

- [x] 在 `notebook/C7 高级 RAG 技巧/README.md` 增加 Recall@k、RR、MRR 的最小定义和算例。
- [x] 更新 `docs/README.md` 的先修表，使 C5 只承担基础评估流程，指标定义链接到 C7 首页。
- [x] 将表示本教程内部章节的 `C2`、`C3` 等写法改为“C7 第 2 章”“C7 第 3 章”。
- [x] 用 `rg -n 'C2|C3|C4|C5' README.md docs/README.md 'notebook/C7 高级 RAG 技巧' -g '*.md'` 检查剩余歧义。

## Task 2: 按执行阶段重组第 6 章导航

- [x] 重写 `notebook/C7 高级 RAG 技巧/6. 处理信息缺口/README.md` 的方法导航，区分索引时、检索前、检索后和会话/来源编排。
- [x] 明确 Late Chunking 在索引时改变向量，并在 `notebook/C7 高级 RAG 技巧/3. 索引阶段/README.md` 增加跨章入口。
- [x] 区分检索前的问题分解与检索后的补查，并说明它们与第 4 章查询改写的关系。
- [x] 压缩 README 中面向维护者的实现细节，保留无 fallback、资料边界和真实结果约束，并链接维护说明与 Notebook。

## Task 3: 修复方法命名、跨章链接和评估练习

- [x] 将 PRCA/REPLUG 的展示名称改为“检索与生成的训练式对齐”，保持 Notebook 文件路径不变。
- [x] 从 `notebook/C7 高级 RAG 技巧/5. 生成阶段/README.md` 增加到 PRCA/REPLUG 的可选跨章链接。
- [x] 修改 `notebook/C7 高级 RAG 技巧/7. 评估/README.md`，要求保存全部真实逐题结果，但不强制实验必须出现不变或退化。
- [x] 精简第 2 章和根入口的审计叙述，保留损失函数知识、数据来源、真实实验结果与适用边界。

## Task 4: 验证教程和发布范围

- [x] 运行 `python -m pytest -q tests/c7`，预期所有 C7 测试通过。
- [x] 运行 `python 'notebook/C7 高级 RAG 技巧/scripts/check_tutorial.py'`，预期 Notebook、链接和教程契约检查通过。
- [x] 运行数据检查与完整项目检查，预期 canonical 数据和四项任务均通过。
- [x] 运行 `git diff --check` 与发布快照检查，只提交本次 C7 信息架构相关文件。

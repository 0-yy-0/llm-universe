# C7 上下文增强：检索命中与 ID 映射修复设计

**日期**：2026-04-11  
**范围**：`notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb`  
**前置**：`docs/superpowers/specs/2026-04-10-c7-context-enhancement-teaching-design.md`（教学优先、共用 `QA_INDICES`）

## 1. 问题陈述（Review 结论）

多次代码审查后，确认以下**实现层**问题（非「算法方向错误」）：

1. **Sentence Window**：用 `txt in hit_texts` 在 `sentence_map` 里反查下标，依赖 Chroma 返回的 `page_content` 与建库字符串**字节级一致**；任何归一化或格式差异会导致**静默丢命中**，窗口变窄或错位。
2. **Small-to-Big / AutoMerging**：用 `hit_set = {page_content}` + `enumerate(child_texts)` 反查 `child_id`，存在**相同脆弱性**；且与 Sentence Window 同源。
3. **AutoMerging（次要）**：`merged_parents` 遍历顺序依赖 `leaf_per_parent` 字典序，**不保证全书阅读顺序**；`sparse_parts[:4]` 与 `parts[:8]` 为硬截断，可能裁掉相关片段（教学上可保留截断但需在文中或注释说明）。

**非目标**：不引入 LangChain `ParentDocumentRetriever` 等高层封装；**保留手写映射**（`neighbor_map`、`child_to_parent`、`leaf_per_parent`），读者仍能看清「检什么、如何扩成上下文」。

## 2. 设计原则

- **显式 ID**：建索引时写入 `metadata`，检索后**优先**用 `metadata` 解析 `sentence_id` / `child_id`（及 STB 所需的 `parent_id` 可冗余写入便于调试）。
- **手写逻辑不变**：扩窗仍用 `neighbor_map`；父块仍用 `child_to_parent` + `parent_texts`；AutoMerging 仍用 `leaf_per_parent` + 阈值。
- **可复现迁移**：向量库若与旧版「纯文本、无 ID」不兼容，** bump 持久化目录名**（或文档明确要求删除旧目录），避免静默加载旧索引导致 metadata 为空。

## 3. 方案对比（简）

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A（推荐） | `Document` + `metadata` + 新 persist 目录 | 可靠、仍手写、教学清晰 | 需重建 Chroma |
| B | 保留旧库，运行时 `page_content` 反查作 fallback | 旧目录可继续用 | 双路径难维护，易掩盖坏索引 |
| C | 仅改 STB、不动 Sentence Window | 改动小 | 半套修复，审查问题仍在 SW |

**选定：A**。

## 4. Sentence Window 改动

- **建库**：不再仅用 `Chroma.from_texts(list(sentence_map.values()))`。改为 `Document(page_content=s, metadata={"sentence_id": i})` 列表，`i` 为与 `sentence_map` / `neighbor_map` 一致的全局句下标（0..N-1）。
- **检索后**：`hit_ids = [int(d.metadata["sentence_id"]) for d in hits]`，并对缺失或非数字 metadata 做**显式告警或跳过**（可选 `print` 一行，避免静默）。
- **窗口拼接**：保持 `window_ids = sorted({... neighbor_map ...})` 与现有 `sentence_map[i]` 拼接逻辑。
- **inspect**：与 `sentence_window_answer` 使用同一套 `hit_ids` 解析规则。
- **持久化目录**：与 notebook 内常量一致，固定为 `sentence_window` / `small_to_big` 等**单一目录名**；改参数时删该目录重建即可，不保留多版历史路径。

## 5. Small-to-Big 与 AutoMerging 改动（共用子块向量库）

- **建库**：每个子块对应 `Document(page_content=child_texts[c_idx], metadata={"child_id": c_idx, "parent_id": p_idx})`。
- **Chroma `ids`**（若 API 支持）：使用稳定字符串 id，如 `f"child-{c_idx}"`，避免重复写入。
- **检索后**：`hit_ids = {int(d.metadata["child_id"]) for d in hits}`（或列表去重），**禁止**再用全文集合反查作为主路径；可删除或仅保留 debug 分支。
- **`small_to_big_answer`**：`parent_ids` 仍由 `child_to_parent[i]` 推导；父块拼接上限等参数保持现有教学设定，除非另有 spec。
- **`auto_merge_answer`**：逻辑不变；可选增强（同轮实现或后续）：对 `merged_parents` 按「该父块下最小 `child_id`」或 `parent_idx` 排序，使上下文顺序更接近文档流；在代码注释或小节 markdown 中说明 `sparse_parts[:4]` / `parts[:8]` 为 token 启发式上限。
- **持久化目录**：固定 `small_to_big`（与上同理）。
- **inspect_small_to_big / inspect_auto_merging**：用 metadata 解析 `child_id`，与 answer 路径一致。

## 6. 共用工具函数（建议）

在 notebook 首段工具区增加小型辅助（命名与现有风格一致即可）：

- `documents_to_chroma(documents, embeddings, persist_directory)`：若目录非空则加载，否则 `from_documents`；避免重复实现三套分支。
- 或分别保留 `build_chroma_from_docs`，但**扩展**为仅接受 `List[Document]` 且要求含所需 metadata（Sentence / Child 两套调用点）。

## 7. 与 Baseline 的关系

- **不强制**在本轮统一 Baseline 与各方法的 `k`（教学 spec 未要求严格 ablation）；若修改 `k`，仅在对比说明中一句话交代即可。
- Baseline 路径**可不动**；若未来也要避免「块内容反查」，再单独立项。

## 8. 验收标准

1. Sentence Window / STB / AM 的 **answer 与 inspect** 均通过 **metadata** 得到 `hit_ids`，主路径无 `page_content in set` 反查。
2. 删除新 persist 目录后全本执行：能创建索引并完成 `qna_dict` 评估；对比表与 `WIN_TARGETS` 打印行为与改前一致或按教学意图微调文案。
3. 教学叙述：`clean_text`、硬截断、与 Baseline 参数差异等**限制**仍在正文或注释中可见，不与「ID 修复」矛盾。

## 9. 实现后自检（Review Checklist）

- [ ] 新 persist 目录名已 bump，旧目录说明已写入 notebook 提示。
- [ ] 无 metadata 的 Document 不会静默进入空 `hit_ids`（至少日志可见）。
- [ ] `compare_df` / 严格胜场 / 回退统计单元格无需改逻辑（除非列名变化）。

## 10. 审批

- **用户**：审阅本 spec 后确认可进入 `writing-plans` / 直接改 notebook 实现。

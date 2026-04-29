# C7 6.1 上下文增强教程优化设计

## 核心目标

重构 `notebook/C7 高级 RAG 技巧/6. 增强阶段/1. 上下文增强.ipynb` 的叙事与实验设计，让 Sentence Window、Small-to-Big、AutoMerging 都作为改进 baseline 的上下文增强方法呈现。

最终教程要让读者清楚看到：baseline 的上下文不足在哪里；每种增强方法补回了什么；为什么这种补回能改善回答；什么题型适合用哪种方法。编写过程中可以使用更多中间表、trace 和参数实验来辅助调试，但最终 notebook 要精简、清晰、可跑通。

## 设计原则

- 不把任何已选增强方法写成失败案例或反例。
- 保留自然主集，避免只挑有利样例。
- 新增方法优势样例集，突出不同方法各自优于 baseline 的场景。
- 最好三种方法在优势样例集上都明显超过 baseline；自然主集用于观察总体趋势和适用边界。
- 共用展示、汇总、保存逻辑抽成函数，避免 notebook 结构散乱。
- 修复当前 CSV 落盘错误，保证 notebook 完整跑通。

## Notebook 主线

采用双层评测叙事。

### 第一层：自然主集

保留现有 24 题作为自然主集，用于观察整体趋势。展示内容包括：

- Baseline、Sentence Window、Small-to-Big、AutoMerging 的同题得分；
- 各方法平均分；
- 各方法相对 baseline 的严格提分题数、回退题数、持平题数；
- 适合进一步分析的代表性题目。

自然主集不承担“每种方法必须总均值都赢”的叙事压力。它用于告诉读者：在真实混合问题上，不同增强方法会命中不同优势题型。

### 第二层：方法优势样例集

新增 `method_advantage_cases`，每种方法 2-3 道代表题：

- Sentence Window：邻句补全型题；
- Small-to-Big：父段落回填型题；
- AutoMerging：同一父块下多个子块命中、需要自动合并的题。

每个优势样例按照统一模板展示：

1. baseline 缺了什么；
2. 增强方法补回什么；
3. 为什么补回内容能帮助回答；
4. baseline 得分与方法得分；
5. 该方法适用的问题类型和额外成本。

## 三种方法定位

### Sentence Window

定位：邻句补全型改进。

保留句级检索和 `WINDOW_SIZE=2` 的主实现。优势样例选择那些 baseline 命中相关位置，但关键定义、原因或例子落在邻句中的问题。

展示字段包括：

- `baseline_context`；
- `sentence_window_context`；
- `added_by_window`；
- `why_it_helps`。

教学重点：固定 chunk 可能切断连续语义；句窗用较低成本补回命中句前后的支撑信息。

### Small-to-Big

定位：小块定位 + 大块生成型改进。

保留父子块结构。优势样例选择那些子块可以精准定位主题，但答案需要父段落里完整定义、公式或解释的问题。

展示字段包括：

- `hit_child`；
- `returned_parent`；
- `baseline_missing`；
- `parent_added_evidence`；
- `why_it_helps`。

教学重点：检索要准，所以用小块；回答要完整，所以回填父块。它适合段落结构清晰的 PDF、技术文档和论文。

### AutoMerging

定位：多子块命中同一父块时的自动合并改进。

AutoMerging 不写成失败或反例。它要展示的是：当同一父块内多个子证据被命中时，系统可以根据命中密度自动决定是否抬升整父块，而不是固定返回碎片或固定返回大块。

调整方式：

- 新增 AutoMerging 专项优势题；
- 在编写阶段做小规模参数 sweep：
  - `MERGE_THRESHOLD = [0.15, 0.25, 0.30]`；
  - `MIN_CHILD_HITS_FOR_MERGE = [1, 2]`；
- 选择在优势样例上稳定优于 baseline、且上下文长度不过度膨胀的默认参数；
- 在最终 notebook 中保留简洁的 `auto_merge_trace_df`，解释合并过程。

trace 字段包括：

- `question`；
- `hit_parent_count`；
- `merged_parent_count`；
- `sparse_child_count`；
- `context_chars`；
- `baseline_score`；
- `auto_merge_score`；
- `delta`。

教学重点：AutoMerging 的价值不是“固定多给上下文”，而是根据同父命中密度在完整性和噪声之间做自动折中。

## 共用函数设计

优先在本 notebook 中抽出共用函数。如果后续 6.2/6.3 也复用，再考虑迁入 `_common.py`。

建议新增：

```python
def score_summary(compare_df, methods, baseline_col="baseline"):
    """返回均值、提分题数、回退题数、持平题数。"""
```

```python
def context_delta_summary(baseline_ctx, method_ctx):
    """对比上下文长度、补回内容片段、额外成本。"""
```

```python
def build_advantage_case_table(cases, baseline_context_fn, method_context_fns, score_lookup):
    """构造方法优势样例展示表。"""
```

```python
def save_context_enhance_outputs(compare_df, summary, out_dir="data"):
    """统一保存主集结果与汇总。"""
```

```python
def auto_merge_trace(question, hit_ids, parent_stats, final_context, scores):
    """返回 AutoMerging 合并轨迹。"""
```

这些函数服务于两个目标：教程结构更清楚；后续调整参数或样例时不用到处改重复代码。

## 指标设计

保留并扩展指标：

| 指标 | 作用 |
| --- | --- |
| `mean_score_0_2` | 自然主集整体趋势 |
| `strict_wins_vs_baseline` | 严格超过 baseline 的题数 |
| `regressions_vs_baseline` | 低于 baseline 的题数，用于调参观察 |
| `advantage_case_score` | 方法优势样例集上的效果 |
| `context_chars` | 上下文长度成本 |
| `added_context_chars` | 增强多带回的上下文 |
| `merged_parent_count` | AutoMerging 是否触发合并 |

结论表达：在自然主集上观察总体趋势，在优势样例集上验证机制有效性。两者共同说明上下文增强如何改进 baseline。

## 产物策略

编写和调试阶段可以生成临时表、参数 sweep 结果、优势样例明细和 AutoMerging trace 文件，辅助选择样例与参数。

最终教程版本只保留最小必要落盘文件：

```text
data/
  context_enhance_compare_table.csv
  context_enhance_eval_summary.json
```

优势样例和 AutoMerging trace 在 notebook 中展示即可，不作为正式教程产物落盘。

CSV 保存统一使用安全参数，修复当前 `need to escape, but no escapechar set` 错误。可采用 `quoting=csv.QUOTE_ALL` 或指定 `escapechar="\\"`。

## 读者体验

每个方法按相同节奏组织：

1. Baseline 在这种题型上缺什么；
2. 该方法如何补；
3. 跑同题评测；
4. 展示优势样例；
5. 总结适用场景与成本。

最终结论写成方法选择指南：

- Sentence Window 改进邻句缺失；
- Small-to-Big 改进段落支撑不足；
- AutoMerging 改进同父多证据碎片化；
- 生产场景可按问题类型路由或组合使用。

## 非目标

- 不为了让表格好看而人为弱化 baseline。
- 不把 AutoMerging 写成反例或失败案例。
- 不把临时调试产物都保留到最终教程。
- 不在本次设计中扩展到流程增强或系统增强章节。

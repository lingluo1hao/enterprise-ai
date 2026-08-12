# Bad Case #9「产品有4种工作模式分别是什么」答非所问 — 根因诊断报告

- 诊断时间：2026-08-11
- 诊断方式：沙箱 anaconda py310 直连生产 VM（Milvus 19530 / Ollama bge-m3 11434 / reranker 11436 / DeepSeek），驱动**真实 app 代码路径**，非静态代码走读
- 结论：**检索链路没有问题，问题出在「检索结果返回给上游」这一步内容被偷换 + 该内容早在入库时已被截断**

---

## 一、一句话结论

命中的子片段是**完全正确**的（RRF 排名 #1，明文含答案），但 `_parse_hits` 把返回给上游的
`Document.page_content` 从「410 字的子片段」换成了「5528 字的整章父文本」，
而这份父文本在**入库时已被按 8192 字节硬截断**，答案位于第 5914 字、截断点在 5528 字，
**答案在写库那一刻就被物理删掉了**。

于是：检索对了 → 返回的内容里没有答案 → 精排在没有答案的文本上打分 → LLM 只能回答"资料中未具体说明"。

---

## 二、决定性证据：A/B 对照实验

同一 query、同一索引、同一 reranker、同一 DeepSeek，**唯一变量 = `page_content` 取 `parent_content` 还是取子片段 `content`**。

| 指标 | 模式 A（现状：parent_content） | 模式 B（修复：子片段 content） |
|---|---|---|
| RRF 候选 20 条 → `[:80]` 去重后 | **13 条**（坍缩 35%） | 18 条 |
| reranker top5 相关性分 | **全为负** `[-3.72, -4.01, -4.33, -4.83, -5.24]` | **正分** `[3.45, 2.17, -1.90, -2.85, -3.13]` |
| 最终 context | 2908 字，**不含答案** | 1832 字，**含答案** |
| DeepSeek 答案 | 「工作模式编号取值1-4，但资料中未具体说明每种工作模式分别是什么」 | 「模式1 实时定位 / 模式2 省电 / 模式3 基站定位 / 模式4 短信」 |
| 四模式命中 | 0 / 4 | **4 / 4** |

模式 A 的输出与线上截图的错误答案**逐字复现**，确认根因定位正确。

### 检索层本身是健康的

```
dense hits: 40 | sparse hits: 40
[dense] answer child at rank #2: ci=14 content_len=410  child content HAS answer = True
[sparse] answer child at rank #1: ci=14 content_len=410  child content HAS answer = True
RRF #1 = ci=14  CHILD_has=True  SERVED_has=False  child_len=410  served_len=5528
```

`CHILD_has=True / SERVED_has=False` 就是本次事故的全部真相。

---

## 三、逐条裁决

### 🔴 疑点 1（成立，且是唯一根因）：`_parse_hits` 把子片段内容换成整章父文本

`advanced_rag_agent.py:721`

```python
page_content = entity.get("parent_content") or entity.get("content", "")
```

命中的是子片段（`is_parent == false`，检索侧已过滤），但返回的正文是整章。
实测 top10 的 `page_content` 长度为 5528 / 5117 / 6042 / 4796 / 6266 字，全是整章。

**连锁灾难 A — RRF 跨 query 去重坍缩**（`langgraph_rag_agent.py:1722`、`:1701` 两处）

```python
key = doc.page_content[:80]
```

同章所有子 chunk 共享同一份 `parent_content`，前 80 字完全相同 → 同章只活下来 1 条。
实测：top10 只剩 **6** 个 distinct，top20 只剩 **13** 个。多路召回与宽候选池（`RETRIEVE_CANDIDATE_K=20`）的收益被这一行吃掉大半。

**连锁灾难 B — 生成侧预算截断错位**（`langgraph_rag_agent.py:1924`）

```python
body = _clean_body(doc.page_content[:trunc])   # RANK_BUDGET[0] = 1200
```

对 410 字的子片段，1200 预算绰绰有余；对 5528 字的整章，只截到前 1200 字，
命中点位在章中/章末的一律被切掉。这正是"工作模式/心跳/通信协议张冠李戴"的直接来源。

### 🔴 疑点 5（本次新发现，你没列到，是疑点 1 的致命放大器）：父文本入库即被截断

```
[原章] 完整长度=7882 字 / 12074 字节，答案 offset=5914 字
       8192 字节截断点 ≈ 5528 字 → 答案在截断点之后？True
实测入库值：parent_content len(chars)=5528  bytes=8192  HAS answer = False
```

`ingest/store.py` 的 `_trunc_bytes` 把 `parent_content` 按 Milvus VARCHAR 上限 8192 **字节**截断。
该章 12074 字节，超限 47%，答案恰好落在被砍掉的那 47% 里。

**这意味着：即便修好疑点 1 的截断错位、把 RANK_BUDGET 调到 8000，答案也永远不会出现**——
因为它根本没被存进 `parent_content`。small-to-big 的父窗口方案对**长章节静默失效**，且没有任何告警。

### 🟠 疑点 2（不成立）：BM25 稀疏检索并没有静默回退

实测反证，三重：

1. 真实调用 `similarity_search_with_score` 并捕获 stdout：**没有任何 `⚠ BM25 稀疏召回失败` 输出**
2. 直接调 `anns_field="sparse"` 传 query 原文：**返回 40 条命中**
3. 答案子片段在 sparse 路排 **#1**（BM25 关键词匹配正常工作）

原因：Milvus 2.5 的 BM25 Function 是**服务端**分词/转稀疏向量的，客户端直接传原始文本就是正确用法，
不需要客户端侧生成稀疏向量。`vs.hybrid = True`，混合检索一直是两条腿在跑。

**这条不用改，改了反而会引入 bug。**

### 🟡 疑点 3（成立，是疑点 1 的派生）：cross-encoder 在整章文本上打分

`langgraph_rag_agent.py:1757` 截断 3000 字送 reranker，而输入正是被换过的父文本。
量化后果见上表：模式 A 的 reranker 分数**全线为负**（-3.72 ~ -5.24，即模型判定"全都不相关"），
模式 B 立刻变成 +3.45。同一个 reranker、同一个 query，仅因为打分对象从整章换成子片段，
判别力从"全盘否定"恢复到"精准命中"。

修好疑点 1 后此条自动消失，**无需单独改动**。

### 🟢 疑点 4（部分成立，属优化项，不影响本 case）

| 项 | 现状 | 判断 |
|---|---|---|
| `RERANK_TIMEOUT=180`（.env） | 叠加 `RERANK_RETRIES` 最坏可卡 6 分钟，而 Gunicorn 120s 先断 | **成立**，建议降到 30~45s |
| `GRADE_THRESHOLD=1` | 只要 1 条相关就停止多轮检索 | **成立但低危**，本 case 中检索本就命中 #1，不是瓶颈 |
| 章节标题前缀只进 content、BM25 帮不上 | `pipeline.py:187` | **表述需修正**：前缀进了 content，而 content 正是 BM25 稀疏索引的取值来源，BM25 是能吃到前缀的；真正吃不到的是被换成 parent_content 后的**下游**环节 |

---

## 四、修复方案（待确认后执行）

### P0 — 一行改动，解决全部 4 条连锁问题

`advanced_rag_agent.py:721`

```python
# 现状
page_content = entity.get("parent_content") or entity.get("content", "")

# 建议
page_content = entity.get("content", "") or entity.get("parent_content", "")
```

同时把父窗口放进 metadata 供生成侧**按需**取用，而不是强行顶替正文：

```python
meta["parent_content"] = entity.get("parent_content", "") or ""
```

这样：reranker 对子片段打分（判别力恢复）、RANK_BUDGET 1200 字对 410 字子片段完全够用（不再错位）。

> ⚠️ **本节原判断已被实测推翻**：这里原写"去重按子片段前 80 字（各不相同，不再坍缩）"。
> 实施后回归发现**并不成立**——见下方「疑点 6」。P0 必须配套身份去重一起改，单改 P0 仍答错。

### 🔴 疑点 6（实施 P0 后回归才暴露，报告初版未预见）：去重 key 用内容前缀，在模板化文档上误杀

改完 P0 重跑回归，`page_content` 已正确变成 410 字子片段、答案在检索层排 **#1**，
但 DeepSeek 仍只答出 3/4（缺"模式1 实时定位"）。打印全量分数发现：

```
fused: 18 | 含答案位次: []
```

**答案在 `_rrf_fuse_queries` 阶段被丢弃了**，根本没进 reranker。

根因：`key = doc.page_content[:80]`。含答案的 `ci=14` 与无关的 `ci=13`，前 80 字**逐字相同**：

```
37 字章节标题前缀（pipeline.py 拼接）+ 43 字模板协议头 "V4,CMD,seq,HHMMSS,S,latitude..."
```

技术文档里同章节的协议报文段天然共享固定头部，前缀去重必然碰撞。
更糟的是 `docs[key] = doc` 是**后者覆盖前者**——rank #16 的无关分片直接顶掉了 rank #1 的答案。

即：P0 只是把"整章前 80 字全同"降级为"同章模板段前 80 字全同"，坍缩规模变小但**照样吃掉答案**。

**修复：去重从"内容相似"改为"身份唯一"**（新增 `advanced_rag_agent._doc_key`），按可靠性降序：

| 优先级 | key | 说明 |
|---|---|---|
| 1 | `pk:{Milvus 主键}` | `_parse_hits` 新增透传 `_pk = hit.id`，绝对唯一 |
| 2 | `{source}#{parent_id}#{chunk_index}` | **必须带 parent_id** |
| 3 | `h:{全文 md5}` | metadata 缺失兜底，用全文不用前缀 |

优先级 2 有个坑：`chunk_index` 是**章节内序号**（`ingest/chunk.py:466`，每章从 0 重新计数），
跨章节会重复。初版只用 `source+chunk_index`，回归时融合从 20 坍缩到 **13**（比旧前缀版的 18 还狠），
补上 `parent_id`（章节指纹）后才恢复 20 → 20 无损。

同时把 `docs[key] = doc` 改为 `docs.setdefault(key, doc)`，同一分片被多 query 召回时保留**首次**（排名更靠前）那份。

**涉及 4 处去重点，全部统一到 `_doc_key`**：

| 文件 | 位置 | 原 key |
|---|---|---|
| `langgraph_rag_agent.py` | `_rrf_fuse_queries` | `page_content[:80]` |
| `langgraph_rag_agent.py` | figure 合并 ×2 | `page_content[:80]` |
| `advanced_rag_agent.py` | `DocSearchSkill` 补搜 ×2 | `page_content[:50]` |
| `advanced_rag_agent.py` | `_retrieve` 多 query 合并 | `page_content[:50]` |

### 疑点 7 — 「生成侧没消费 parent_content」是不是 #9 的根因？

**不是。** 这是个因果误判，必须和 #9 的真正死因分开：

- 改 P0 之前，生成侧其实**一直在消费** `parent_content`——只不过是通过错误通道
  （`_parse_hits` 把 `parent_content` 塞进了 `page_content` 主干道）。那时它"消费"了，可 #9 照样错。
  所以"没消费"不可能是原 bug 的原因。
- 答案在 `_rrf_fuse_queries` 上游就被去重吃掉（`fused: 18 | 含答案位次: []`），
  根本到不了生成节点。生成侧怎么读 `parent_content` 都救不回来。
- P0 + 身份去重修完后，生成侧只读子片段（**完全没碰** `parent_content`），#9 实测 4/4。
  修复根本不依赖生成消费父窗口。

结论：原 bug 是**检索出口契约错 + 去重碰撞**（上游），与生成侧读不读父窗口无关。
"生成侧没消费 parent_content" 是 P0 修复后**新暴露的独立问题**——small-to-big 的 big 半边
被孤立在 metadata 里没人取，导致长答案上下文不足。它影响的是**上下文丰富度**，不是 #9 的答非所问。

### Plan A — 生成侧消费 parent_content（已落地，2026-08-11）

与 P1 解耦、不动数据、不重建索引、短章节立刻受益：

- 抽取模块级 `_build_context(query, docs)`：`RANK_BUDGET` 分级预算 + B3 排序 + 父窗口回填集中一处，
  便于检索层回归直接调用（无需构造整个 RAG App）。
- 父窗口回填设计点（均经探针实测）：
  1. **尾部锚点** `child[-80:]` 是唯一可靠定位点：pipeline 给 child 前加了 37 字章节标题前缀，
     头部不可靠；尾部纯净正文必在 parent 中（除非父被 8192B 截断）。
  2. **只向后扩**：向前会撞章节前缀已覆盖区造成重复；答案延续在命中点之后。
  3. **`parent_id` 去重 + 双重预算 + 降级**：同章只补一次（否则同章 N 条拼 N 遍爆 context）；
     单次 `PARENT_EXPAND_RADIUS=400` / 总量 `PARENT_EXPAND_TOTAL=1600` 双封顶；
     定位失败（`pc.rfind(anchor)` 失败，父截断/前缀异常）静默降级为纯子片段。
- 两个实现细节：`used = 0` 循环外初始化；`pc.find(anchor)` → `pc.rfind(anchor)`
  （模板化文档协议头重复，取最后一个更准）。
- **实测**：线上多 query 确认生效（基站/短信/心跳/设备确认均带 `[章节续文]`，体量 1652~3657 字 < 4500）；
  #9 因其相关章节（命令集）父窗口已被截到 8192 字节、子片段尾部落在截断点之后 → A 不扩张（符合预期，属 P1 范畴）。
- 回归：`scripts/eval_parent_expand.py`（升级自 `_probe_parent_expand2.py` 的尾部锚点 / 同章覆盖率探针，
  新增确定性合成机制用例 + A 开/关对照 + 降级路径 + 答案层 A/B）。全绿。

### P1 — 修复 small-to-big 的静默截断（**已完成，2026-08-11 实测**）

**性质定位（先讲清，防止误判优先级）**：P1 **不是正确性 bug，是上下文丰富度（detail richness）问题**。
实测已坐实：Plan A 开/关下 Bad Case #9 均 4/4（`A on: 缺=[]` / `A off: 缺=[]`）；命令集父窗口被截到 8192 字节
只影响"答案细节补全的丰富度"，不影响答案正确性。故 P1 不阻塞线上、不与 #9 闭环耦合——它是把 small-to-big 的 big 半边
从"残缺整章"补成"完整局部窗口"的增强项。

**治本方向 = 方案 a（参数已定死，避免上 VM 反复）**：

- **改动位置**：`ingest/chunk.py` 的 `parent_content` 生成逻辑（检索侧零改动）。
- **滑动窗口定义**：对每个子片段，`parent_content = 源章节文本以子片段 span 为中心、前后各 N 字的窗口`
  `[child_start - N : child_end + N]`，**`N = 500~600`**（窗口总长 1000~1200 字）。
  子片段贴近章节边界时向可用文本夹紧（不 wrap、不越界）。
- **字节预算**：窗口 ~1200 中文字 ≈ 3600 UTF-8 字节，**远低于 8192 字节上限**，整章截断问题自然消失；
  不再依赖 `store._trunc_bytes` 兜底（可顺手移除那段静默截断，保留 WARN 日志即可）。
- **连 `content` 一起改**：本批改动**同时重定义 `content` 与 `parent_content` 的生成**，一次性合入同一次 `chunk.py` 编辑；
  `content` 保持"精确子片段"语义不变，但其 **37 字章节标题前缀必须保留**——Plan A 的尾部锚点 `child[-80:]` 依赖它在尾部。
  两者在同一重建里一起落库，避免为改 `content` 再重建一次索引。
- **检索侧零改动 + 须验证**：`_parse_hits` / `_build_context` / `_doc_key` 都不用动；但重建后必须跑
  `scripts/eval_parent_expand.py`（检索层 + 答案层）回归，确认：① 确定性合成机制仍全绿；
  ② 答案层 #9 在 A 开/关下仍均 4/4；③ **命令集 query 现在也应吃到 `[章节续文]` 展开**
  （方案 a 后窗口以子片段为中心，`child[-80:]` 必在 `parent_content` 内，`rfind` 对含命令集在内的所有 chunk 100% 命中）。

**额外收益（方案 a 落地后的连锁简化，可选 v2）**：父窗口变"以子片段为中心的局部窗口"后，Plan A 的尾部锚点 `rfind`
对所有 chunk 恒定命中，可把 `_build_context` 从"尾部锚点单侧后扩"平滑降为"直接拼 `parent_content` 局部窗口"——
big 半边更稳、代码更短。不在本次 P1 范围。

**方案 b（仅缓解，非治本，留作对照）**：保留整章，超限时按「以子片段为中心向两侧扩展至 8192 字节」裁剪，而非从头截断；
并在 `ingest/store.py` 截断处打 WARN。但它**仍受 8192 字节上限约束**（Milvus VARCHAR 单字段约 65535 字节天花板），
命令集这类章节可上万字，即便从中心扩也塞不下整章——长章节答案照样丢。故 b 只能作为过渡，a 才是真根治。

**执行约束（排期要点）**：
- 须**重建索引**；重建在 **anaconda py310（应用真实运行环境）** 跑 `python -m ingest.cli ingest --force`，
  它直接连 VM（`192.168.200.128`）的 Milvus/Ollama，落库路径为 `./knowledge/...` 相对路径，与线上一致。
  （早先担心的"沙箱重建落绝对路径"指的是**托管 workbuddy python 沙箱**，不是 anaconda py310——后者就是 app 本身。）
- **真实坑（已踩已修）**：P1 前残留的旧实体是**绝对路径 `D:\...` 且反斜杠**；新代码 `pipeline.run` 把 `\` 归一成正斜杠，
  导致 `delete_by_file` 的 `file_path == "..."` 精确匹配**永远不匹配反斜杠旧值** → 静默删 0。
  `--force` 重建后旧绝对 `D:\...` 实体成了孤儿重复实体；清理须用 `file_path like "D:%"` 通配删（`ingest/cli delete` 的精确删无效）。
- **合并维护窗口**（仍建议）：重建索引代价高，P1 本可挂在下次真实文档批量 ingest 一起做；本次因已验证、参数定死，
  直接 `--force` 重建（2 个协议 PDF，约 13 分钟 CPU embedding）亦无碍。
- 动工前本 spec 已定死（N 值、content 同改、检索侧验证清单），重建不再反复确认参数。

### P2 — 配置项

- `.env`：`RERANK_TIMEOUT` 180 → 45
- `GRADE_THRESHOLD` 1 → 2（观察 grade 的 LLM 调用量再定）

---

## 五、回归验证方法

1. 改完 P0 后重启 Flask，用原 query 重问，期望答案含四种模式
2. 跑 `python -m evalkit.runner --suite both`，对比改动前后的检索层/答案层通过率
3. 重点回归**图片相关 query**：`_parse_hits` 的 `figure_paths` 与父文本解耦后，
   确认图页召回（`search_figure_pages`）与答案末尾追图逻辑未受影响

---

## 六、附：本次诊断纠正的既往错误结论

| 既往结论 | 纠正 |
|---|---|
| `RETRIEVE_TOP_K=5` 截断导致 gold 丢失 | 对本 case 不成立，k=5 与 k=20 的 top5 完全相同 |
| loader 吞掉答案正文（`ingest/loaders.py` 表格误判） | **确实是一个真 bug 且已修**，答案现已入库；但它不是本 case 最终答错的原因 |
| 2026-08-11 上午"端到端验证通过、#9 已闭环" | **该验证无效**。验证脚本自己取的是 `content` 字段，绕开了 app 真实使用的 `parent_content`，因此测出"答对"而线上仍答错。教训：验证必须走 app 真实代码路径（`VectorStoreManager.similarity_search_with_score`），不能自建等价脚本 |
| 本报告 P0「一行改动，解决全部 4 条连锁问题」 | **不成立**。P0 只解决了 3 条（精排失灵 / 预算错位 / 坍缩规模），去重坍缩并未随之消失，见疑点 6。真正闭环需要 P0 + 身份去重两件套 |

---

## 七、实施结果（2026-08-11 已完成并实测）

### 改动清单

| 文件 | 改动 |
|---|---|
| `advanced_rag_agent.py` | `_parse_hits` 出口契约：`page_content` 返回子片段；父窗口挪进 `metadata["parent_content"]` 旁路；新增透传 `_pk`。新增 `_doc_key()` helper。`DocSearchSkill` / `_retrieve` 共 3 处去重改用 `_doc_key` |
| `langgraph_rag_agent.py` | 导入 `_doc_key`；`_rrf_fuse_queries` + figure 合并共 3 处去重改用 `_doc_key`；`docs[key]=doc` → `setdefault`；**新增模块级 `_build_context`（方案 A 父窗口回填：尾部锚点 `pc.rfind` 定位、只向后扩、parent_id 去重、双重预算、降级）**，`_do_generate` 改为调用它 |
| `ingest/chunk.py` | **P1 方案 a**：新增 `PARENT_WINDOW_CHARS=550` + `_window_around()`；`_split_prebuilt_section`(PDF 主路径) 与 `_split_section`(md/html) 的 `parent_content` 由「整章」改为「子片段前后各 550 字滑动窗口」（天然 <8192、不共享不截断不膨胀）；docstring 同步更新 |
| `scripts/eval_retrieval_bury.py` | 同步出口契约，避免评测绕开线上行为测出"假绿" |
| `scripts/eval_parent_expand.py` | 新增回归（升级 `_probe_parent_expand2.py` 探针 + 确定性合成机制用例 + A 开/关对照 + 降级路径 + 答案层 A/B） |
| `.env` / `README.md` | `RERANK_TIMEOUT` 180 → 45 |

契约说明与踩坑原因均以**长注释形式写在代码里**（`_parse_hits` 出口处、`_doc_key` docstring），
防止后人再次"顺手优化"回去。

### 回归实测（真实链路：`similarity_search_with_score` → RRF → reranker → DeepSeek）

```
QUERY: 产品有4种工作模式分别是什么
[1] 出口 20 条 | distinct key=20 | pk 可用=True
[2] 融合去重 20 -> 20          ← 无假坍缩（改前 20 -> 13）
[3] 精排 top5 | context 1894 字
[4] 命中 4/4: {'实时定位': True, '省电': True, '基站': True, '短信': True}
    答案：模式1 实时定位模式 / 模式2 省电模式 / 模式3 基站定位模式 / 模式4 短信模式

QUERY: 基站信息格式是什么
[1] 出口 20 条 | distinct key=20 | pk 可用=True
[2] 融合去重 20 -> 20
[4] 命中 1/2: {'LBS': False, '基站': True}   ← 关键词表述差异，非 bug
```

**Bad Case #9 闭环。**

### 后续

- **Plan A（生成侧消费 parent_content）已完成并实测**（见上文「Plan A」小节）：长答案上下文补全，不阻塞、不膨胀。
- **P1（父窗口 8192 字节静默截断 / 滑动窗口根治）已完成并实测**：改 `ingest/chunk.py` 的 `parent_content` 生成
  （子片段为中心、前后各 **N=550** 字滑动窗口，连 `content` 一起改，**检索侧零改动**）；在 anaconda py310 跑
  `python -m ingest.cli ingest --force` 重建（2 个协议 PDF → 908 新分片），并清理了 P1 前残留的绝对路径孤儿实体。
  实测：**命令集长章节 chunk 现在吃到 `[章节续文]`（#9 context 含 `[章节续文]=True`，2321 字）**，且 **#9 答案层仍 4/4**（A 开/关对照全中、无回归）；
  检索层 + 答案层回归 `eval_parent_expand.py` 全绿。性质是 detail 丰富度增强，非正确性 bug——已闭环。
- 所有代码改动仅在工作区，**未提交**（git 纪律：AI 不主动 commit/push）。

### 附：同批次顺手修复的关联 bug —— `delete_by_file` 历史路径静默删不掉

P1 重建时暴露：`delete_by_file` 用精确匹配 `file_path == "..."`，但历史实体可能带 `./` 前缀、
反斜杠或绝对路径（旧 Windows 直接写入），精确匹配**永远落空 → 静默删 0 条**，旧实体沦为孤儿。

修复（`ingest/store.py`）：精确匹配落空后兜底改为 **Python 侧归一化匹配 + 按 `chunk_id` 主键删除**
（`_norm_path_key` 把任意形态归一成 `knowledge/子目录/文件名` 稳定 key；`_match_ids_by_path` 分页拉
全量比对）。实测三种历史写法（无 `./` 前缀 / 带 `./` / 反斜杠绝对路径）均精确命中、反例 0 误伤、
一次性测试集合端到端通过。**注意：不能改用 Milvus `LIKE` 兜底——`file_path like "%中文_含(括号).pdf"`
会直接报 1100 表达式解析失败（`_`/`(` 触发解析器异常），必须用 Python 侧比对。**

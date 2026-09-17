# SAGE-E baseline 方案（空间类型识别 / 第三个比较项）

> 目标：把 **SAGE-E** 接入 CAD2Graph 的空间语义层，作为 `ISpaceTypeClassifier` 的
> 第三个可插拔实现，与 `LLMMultiStage`、`TextMatching` 并列比较。
>
> 与 `docs/vecfloorseg_baseline_plan.md` 不同，本文档**所有结论均为实测**，而非从
> README 转述。文中标注 ✅ 的条目都附了验证方式。

---

## 1. 结论先行

| 维度 | 结论 |
|---|---|
| 任务匹配 | ✅ **完全对口** —— SAGE-E 做的就是 room type classification，即 CAD2Graph 的 `ISpaceTypeClassifier` |
| 权重可用性 | ✅ **开源可用**（68 KB，MIT），**无需训练即可推理** —— 这是 VecFloorSeg 完全没有的优势 |
| 公开数据集 | ✅ **有**：`RoomGraph`（896 KB，224 套户型 / 2076 个房间），MIT，随仓库发布 |
| 环境代价 | 🟢 **可完全避开 DGL** —— 模型只有 4 层、15,394 参数，前向可 15 行 numpy 等价实现 |
| 训练成本 | 🟢 论文配置 **200 epoch 在 CPU 上 391 秒**；即使重训也是分钟级，**不需要 GPU** |
| 主要风险 | 🟡 8 维节点 / 5 维边特征的**精确语义未公开**；类别体系 9 vs 21 需要对齐层 |
| 综合可行性 | 🟢 **高**，预计 **2.5–3.5 天** |

**比较价值**：SAGE-E **完全不读图纸文字**，纯几何 + 拓扑推理，正好补上现有两个分类器之间的空档。

| 分类器 | 文字 | 大模型 | 几何/拓扑 | 需要训练 |
|---|---|---|---|---|
| `LLMMultiStage` | ✅ | ✅ | ✅ | ❌ |
| `TextMatching` | ✅ | ❌ | ❌ | ❌ |
| **`SAGEE`** | ❌ | ❌ | ✅ | ✅ |

> 注册名是 `SAGEE`（别名 `SAGE` / `SAGE_E`），不是 `SAGEndToEnd`。
> 完整的两任务基准协议已独立成文：**[benchmark_protocol.md](benchmark_protocol.md)**。

---

## 2. 来源与事实

* 代码：<https://github.com/ZijianWang-ZW/SAGE-E>（MIT，2021-10 创建，2025-05 最后推送）
* 论文（期刊，本方案对应的版本）：Wang, Z., Sacks, R., Yeung, T., 2022.
  *Exploring graph neural networks for semantic enrichment: Room type classification.*
  Automation in Construction 134:104039. DOI `10.1016/j.autcon.2021.104039`（Thorpe Medal）
* 免费会议论文（前身，7 类版本）：
  <https://cbim2020.net.technion.ac.il/files/2021/10/w78-2021-paper-077.pdf>
* 作者主页：<https://zijianwang-zw.github.io/>

仓库只有 5 个文件，**没有任何特征构造代码**：

```
code/SAGEE.py            1829 B   模型定义（4 层 SAGE-E）
code/node_evaluation.py  1587 B   评估工具
code/train&test.ipynb   38366 B   训练+测试 notebook
code/best_default.pt    68847 B   预训练权重
code/best_user.pt       68341 B   预训练权重（调参版）
dataset/roomgraph.bin  896344 B   RoomGraph 数据集
```

---

## 3. 实测数据契约

以下全部由 **实际读取权重与数据集** 得到，不是推测。

### 3.1 模型（读 `best_default.pt` 的 state_dict）

```
4 层 SAGEELayer, 隐藏维 (50, 50, 25, 9)
  layer 0: W_msg(50,13)  W_apply(50,58)    -> ndim_in=8  edim=5  ndim_out=50
  layer 1: W_msg(50,55)  W_apply(50,100)   -> ndim_in=50           ndim_out=50
  layer 2: W_msg(25,55)  W_apply(25,75)    -> ndim_in=50           ndim_out=25
  layer 3: W_msg( 9,30)  W_apply( 9,34)    -> ndim_in=25           ndim_out=9
总参数量: 15,394
```

* **8 维节点特征 / 5 维边特征 / 9 类 / 4 层**
* dropout `p=0.2`（无参数，故不出现在 state_dict）
* 权重是 `pickle` 出来的 `SAGEE` 对象；用 5 行的空壳 `SAGEE.py` 即可 `torch.load()`，
  **说明 `dgl` 只在 `forward()` 里用**

### 3.2 前向传播（读 `SAGEE.py` 源码，逐行）

```python
for i, layer in enumerate(self.layers):
    if i != 0:
        nfeats = self.dropout(nfeats)
    nfeats = layer(g, nfeats, efeats)      # 见下
```

单层内部：

```python
m = sum_{边 u->v} ReLU(W_msg @ [h_u ‖ e_uv])      # 消息 + 求和聚合
h_v = ReLU(W_apply @ [h_v ‖ m])                    # 更新
```

即

$$m_v=\sum_{u\to v}\mathrm{ReLU}\big(W_{msg}\,[h_u\;\|\;e_{uv}]+b_{msg}\big),\qquad
h_v\leftarrow\mathrm{ReLU}\big(W_{apply}\,[h_v\;\|\;m_v]+b_{apply}\big)$$

⚠️ DGL 的 `update_all(..., fn.sum(...))` 语义是**对入边求和**，且 notebook 里
`add_self_loop` 是**注释掉的** —— 端口实现必须保持"无自环"。

### 3.3 数据集（读 `roomgraph.bin`）

```
图数量      : 224
节点(房间)  : 2076   每图 min/median/max = 5 / 9 / 13
边          : 6634   每图 min/median/max = 10 / 30 / 48

ndata['feat']     (2076, 8)  float64
ndata['label']    (2076, 9)  int64   one-hot（已校验每行只有 1 个 1）
ndata['index']    (2076,)    int64
edata['relation'] (6634, 5)  int64
```

**8 维节点特征实测统计：**

| 列 | 类型 | 范围 | 推断语义 | 依据 |
|---|---|---|---|---|
| f0 | 二值 | {0,1}，0 有 269 个 | 布尔标志 | 只有 2 个取值 |
| f1 | 二值 | {0,1}，0 有 514 个 | 布尔标志 | 同上 |
| f2 | 二值 | {0,1}，0 有 1436 个 | 布尔标志，把 9 类干净地切成两组 | 类 {0,1,2,8} 均值 ≈ 1，类 {3,4,5,6,7} 均值 = 0 |
| f3 | 计数 | 0–9，长尾 | **邻接房间数** | 与图内度数 corr = **+0.817**，且恒 ≤ 度数 |
| f4 | 计数 | 0–5，mean 1.21 | **门数 / 去重邻接数** | 小整数长尾，0:514 1:830 2:544 3:168 4:19 5:1 |
| f5 | 连续 | 0.06–22.69，p50 1.43，p95 9.24 | **面积类**（单位待确认） | 602 个不同取值，量级像 m² 但中位数偏小 |
| f6 | 连续 | 0.0–9.9，54 个取值，p50 1.20 | **长度类** | 类似"共享墙长/周长" |
| f7 | 二值 | {0,1}，1 有 221 个 | 稀有布尔标志 | 且与度数 **负相关**（corr −0.268） |

**5 维边特征 —— 实际只有 4 种组合，覆盖 100%：**

| flag 组合 (f0..f4) | 边数 | 占比 |
|---|---|---|
| `0 1 0 0 0` | 2503 | 37.7% |
| `1 0 0 0 1` | 2018 | 30.4% |
| `1 0 0 1 0` | 1151 | 17.4% |
| `0 0 1 0 0` | 962 | 14.5% |

→ 所谓"5 维边特征"其实是 **4 种关系类型的编码**（不是 5 个独立标志，也不是标准 one-hot）。
会议论文明确只有 3 种关系（门连接 / 墙连接 / 虚墙连接），期刊版扩展到 4 种。

**9 类标签实测分布与"每图个数"：**

| class | 总数 | 占比 | 每图平均 | 出现图数 | 备注 |
|---|---|---|---|---|---|
| 0 | 223 | 10.7% | 1.00 | 223 / 224 | 几乎每套必有且唯一 |
| 1 | 224 | 10.8% | 1.00 | **224 / 224** | **必定存在且唯一**，且 f3=5.03（连通度最高） |
| 2 | 216 | 10.4% | 0.96 | 216 / 224 | 几乎每套必有且唯一 |
| 3 | **416** | **20.0%** | 1.86 | 223 / 224 | **= Bedroom**（论文明确"bedrooms 最多"，且论文混淆矩阵第 3 行和最大） |
| 4 | 162 | 7.8% | 0.72 | 162 / 224 | |
| 5 | 363 | 17.5% | 1.62 | **224 / 224** | 每套必有，常出现 2 个（→ 卫生间/阳台） |
| 6 | 106 | 5.1% | 0.47 | 72 / 224 | 最少，f5 最大（4.85） |
| 7 | 221 | 10.6% | 0.99 | 181 / 224 | 出现即唯一 |
| 8 | 145 | 7.0% | 0.65 | 145 / 224 | |

> 会议论文（7 类）的类别顺序为 KIV, LIV, DIN, BED, TOI, BAL, LAU，且 BED 在第 3 位 ——
> 与这里 **class 3 = Bedroom** 完全吻合。9 类版大概率是"前 7 位沿用 + 追加 2 类"，
> 但这只是**强推测**，不是实测结论，见 §5.1 的确认路径。

### 3.4 论文给出的性能（复现目标）

* 期刊版：test accuracy **0.7970**，macro-F1 **0.7801**
* 会议版（GraphSAGE，仅用关系特征）：**73%**
* 训练配置：`batch_size=1, lr=0.005, weight_decay=5e-4, epochs=200`，CPU 391 秒

### 3.5 ★阶段 A 已完成：复现结果（实测）

用本仓库的**无 DGL 端口**跑作者的权重，与 notebook 打印的混淆矩阵对比如下：

| | 本仓库 | 论文 / notebook |
|---|---|---|
| `best_user.pt` Accuracy | **0.7928** | 0.7970 |
| `best_user.pt` Macro F1 | **0.7801** | **0.7801** ✅ 四位小数一致 |
| `best_default.pt` Accuracy | **0.8169** | — |
| `best_default.pt` Macro F1 | **0.8099** | — |

**`best_user.pt` 的 9×9 混淆矩�阵 81 个格全部与 notebook 逐位相同**：

```
[[35  0  9  0  0  0  0  0  1]      ← 与 notebook 完全一致
 [ 1 44  0  0  0  0  0  0  0]
 [10  0 34  0  0  0  0  0  0]
 [ 0  0  0 72  6  3  0  0  0]
 [ 0  0  0  3 27  2  0  0  0]
 [ 1  0  0 17  0 46  3  0  3]
 [ 0  0  0  2  0 17  8  0  0]
 [ 0  0  0  0  0  0  0 41  0]
 [ 0  0  0  4  0  4  0  0 22]]
```

这同时验证了四件事：**无 DGL 前向语义等价**、**`.bin → .npz` 导出无损**、
**数据划分与 sklearn 逐位一致**、**指标定义正确**。

> 有意思的是 `best_default.pt`（默认超参）比 `best_user.pt`（调参版）在当前划分上
> 高 2.4 个点。论文报的是 0.7970，对应 `best_user.pt`。

其他实测：

* 训练速度：本实现 **200 epoch ≈ 120 秒**（CPU），比 notebook 的 391 秒快约 3 倍 ——
  省掉了逐图 DataLoader 与 DGL 的开销。
* 集成代价：**推理侧可以在 `cadruler` 环境（无 torch、无 DGL）跑出完全相同的混淆矩阵**，
  只需 numpy。实测命令：`python scripts/check_sagee_port.py`。

---

## 4. RoomGraph 能不能合并进我们的数据集？

### 4.1 可以，但只能"图级"合并

✅ **RoomGraph 是公开的、MIT 许可的、随仓库发布的**（`dataset/roomgraph.bin`，896 KB，
224 套户型，其中会议论文明确 **100 套来自中国**、124 套来自英/美）。这是合法的、
可以直接用的数据。

⚠️ **但它只有图，没有原始图纸**：没有平面图、没有坐标、没有几何、没有文字。
所以：

| 合并方式 | 可行性 |
|---|---|
| 重新从图纸提取特征 | ❌ 不可行 —— 源图纸未发布 |
| 按它已固化的 8/5 维特征直接拼图 | ✅ 可行 —— **前提是我们能在 CAD2Graph 侧复现同样的 8 维** |
| 只用它预训练、把 backbone 迁到我们的特征空间 | ❌ 不可行 —— 输入维度必须逐列对齐 |

结论：**合并 = 我们在 CAD2Graph 侧造出与 RoomGraph 逐列同构的 8/5 维特征，然后把两边
的图拼在一起训练。**

### 4.2 合并的可行性判断

✅ **乐观**。对照 §3.3 反推出的特征语义，**每一列 CAD2Graph 都能算出来**：

| RoomGraph 列 | 需要的 CAD2Graph 输入 | 现成来源 |
|---|---|---|
| f3 邻接房间数 | `SpatialContour.neighbors` / `bot:adjacentZone` | ✅ 已有 |
| f4 门数 | 房间之间的门构件 | ✅ 已有（`bot:containsElement` + 门几何） |
| f5 面积类标量 | 轮廓面积 | ✅ 已有（`props:hasArea`） |
| f6 长度类标量 | 轮廓周长 / 共享墙长 | ✅ 已有（几何可算） |
| f0/f1/f2/f7 布尔标志 | 外墙/采光/套内/公共 之类 | ✅ 可由几何 + 图层判定 |
| 边 4 类关系 | 门连接 / 墙连接 / 虚墙连接 / +1 | ✅ 已有（CDT/RGP 的 virtual blocker 概念 + 门几何） |

**合并的规模收益**（实测数字）：

| | 图 | 房间 |
|---|---|---|
| RoomGraph | 224 | 2076 |
| CAD2Graph GT（`output/gt/`，41 份标注图） | 41 | 1951 |
| **合计** | **265** | **≈ 4000** |

房间数翻倍，图数从 41 → 265（**图级多样性提升 6 倍**，这对 GNN 泛化比对房间数更关键）。

### 4.3 合并前必须先确认的事（唯一阻塞项）

**8 列的精确语义未公开。** f5 的单位是多少？f0/f1/f2/f7 分别是什么布尔量？
f6 是周长还是共享墙长？这些决定了我们能否"逐列对齐"。

三条确认路径（按性价比排序）：

1. **直接问作者** —— 作者在主页明确写了欢迎联系（`zijianwang1995@gmail.com`），
   一个邮件就能解决，成本最低。
2. **读期刊论文**（付费墙，DOI `10.1016/j.autcon.2021.104039`）—— 论文必然有特征表。
3. **统计反推** —— 我们已经能确定类型/范围/相关性（§3.3）。若配合 RoomGraph 的
   标签分布与常识（卧室面积应大于卫生间等），可以进一步收敛。

⚠️ **但这不阻塞主流程**：即使不合并，CAD2Graph 自己的 1951 个标注房间也够训一个
15k 参数的模型（RoomGraph 全量才 2076 个）。合并是"锦上添花"，不是"必要条件"。

---

## 5. 类别差异怎么处理

### 5.1 两边的类别体系

| | 类别数 | 说明 |
|---|---|---|
| RoomGraph | **9** | 会议版 7 类：Kitchen / Living / Dining / Bedroom / Toilet / Balcony / Laundry；期刊版扩到 9 |
| CAD2Graph `STANDARD_SPACE_TYPES` | 18 | 见 `src/spatial/classifiers/prompts.py` |
| CAD2Graph GT 实测 | **21** | 见下表 |

✅ **CAD2Graph GT 实测分布**（41 份 `output/gt/*_gt.jsonld`，1951 个空间；
类型写在 `@type` 里，形如 `["bot:Space", "bldg:Bedroom"]`）：

```
Bedroom 368  Balcony 275  Bathroom 226  MainCorridor 160  Kitchen 146
LivingRoom 144  DiningRoom 140  SecondaryCorridor 140  ElevatorShaft 74
PublicCorridor 68  Garden 66  Entrance 52  Stairwell 40  ElectricalRoom 38
StorageRoom 37  StudyRoom 33  WaterRoom 29  Cloakroom 29  VentilationRoom 10
Corridor 6  SunRoom 3
```

**核心矛盾**：CAD2Graph 有大量 RoomGraph 根本没有的类（电梯井、楼梯间、设备间、
公共走廊、花园…），而 RoomGraph 把住宅内部细分得更专（洗衣房等）。

### 5.2 处理方案：三层标签空间 + 分层评估

```
        ┌──────────────────────────────────────────────────────┐
L2 全集 │ 按图纸切分 ── 切片 A ─▶ 评估 SAGE-E v s LLMMultiStage │  → 与 21 类对比
  21 类 │               切片 B ─▶ 评估 SAGE-E v s TextMatching   │
        └──────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────┐
L1 交集 │ 按图纸切分（同一份数据）                              │  → 与 SAGE-E 论文
  9 类  │ 类别映射表把 21 类折叠到 9 类公共空间                 │     的 0.797 对话
        └──────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────┐
L0 本体 │ 显式映射表（写死在代码里、可单测）                    │  → 可审计
        └──────────────────────────────────────────────────────┘
```

**L0 — 本体映射表（必须显式，不能靠假定）**

⚠️ **绝不能假设两边索引一致。** RoomGraph 的 0–8 是它自己的顺序，我们的顺序由
`STANDARD_SPACE_TYPES` 决定。映射表必须：

1. 一边写 `CAD2Graph 类名 → RoomGraph class index`，另一边写反向；
2. 明确标注每个映射的**置信度**（`exact` / `likely` / `unknown`）；
3. 配一个单测，保证 mapping 是双射（或明确记录一对多/未映射项）；
4. 把 §3.3 实测出的"每图平均个数"作为**合理性校验**：例如 Bedroom 在 RoomGraph
   是 1.86/套，在我们 GT 是 368/41 图；若映射后数量级对不上，说明映射错了。

**L1 — 公共 9 类子集评估**

只保留能与 RoomGraph 对齐的类别，其余折叠或排除。这样得到的数字**才能和 SAGE-E
论文的 0.797 / 0.780 放在一起看**。需要排除的典型是：
`ElevatorShaft`、`Stairwell`、`PublicCorridor`、`Garden`、`ElectricalRoom`、
`VentilationRoom`、`WaterRoom` —— 这些在公寓内部图里不存在。

**L2 — 全集 21 类评估**

按图纸切分，用同一份训练/测试划分，比较 SAGE-E / LLMMultiStage / TextMatching。
这是 CAD2Graph 论文里真正要报的数字。注意 21 类下宏平均会被长尾拖垮
（`SunRoom` 只有 3 个样本），所以必须同时报：
macro-F1、weighted-F1、top-3 accuracy、以及混淆矩阵。

**长尾处理规则（L2 用）**

| 情况 | 处理 |
|---|---|
| 样本 < 5（`SunRoom` 3、`Corridor` 6） | 合并到语义最近的父类（`SunRoom`→`Balcony`，`Corridor`→`MainCorridor`）或标 `Other` 并在 macro 里排除，**两种都报** |
| `MainCorridor` / `SecondaryCorridor` / `PublicCorridor` / `Corridor` | 若 L1 需要，统一折叠成 "Corridor" |
| 结构类（电梯/楼梯/设备） | L1 直接排除；L2 保留 |

**合并训练时的额外规则（如果 §4.3 确认成功）**

1. 联合训练只用 **9 类公共空间**（RoomGraph 无 21 类）；
2. 特征必须逐列同构 —— 在拼接前对两边做**逐列归一化**（z-score 或 min-max），
   否则 RoomGraph 的原始量纲会主导；
3. 报告三档结果：
   `仅 CAD2Graph` / `仅 RoomGraph` / `RoomGraph 预训练 + CAD2Graph 微调`；
4. ⚠️ 若两边节点特征分布差异过大，**"先在 RoomGraph 上预训练"可能反而变差** ——
   这是必须实测的，不能假定迁移一定有效。

### 5.3 ★ 关键发现：标签可达性受**轮廓划分**限制（实测）

训练集必须在**系统自己提取的轮廓**上标注（否则训练/推理的输入分布不一致）。
但系统轮廓与 GT 空间并不是同一套划分 —— 这会直接决定"哪些类别根本不可能被预测出来"。

实测（`scripts/inspect_sagee_matching.py`，单张图纸 `2suite (1)` 对 GT）：

| | **CDT** | **RGP** | GT |
|---|---|---|---|
| 系统轮廓数 | 24（0.62×） | **35（0.90×）** | 39 |
| `DiningRoom` 覆盖 | 0 / 2 ❌ | **2 / 2** ✅ | 2 |
| `PublicCorridor` 覆盖 | 0 / 1 ❌ | **1 / 1** ✅ | 1 |
| `SecondaryCorridor` 覆盖 | 0 / 2 ❌ | **4 / 2** ✅ | 2 |
| `Balcony` 覆盖 | 3 / 7（43%） | **7 / 7（100%）** ✅ | 7 |
| `LivingRoom` 面积 | 52.6 m²（吞掉餐厅/玄关/走廊） | **19.7 m² = GT 精确一致** ✅ | 19.7 |

**根因**：CDT 把开敞区域（客厅+餐厅+玄关+走廊）合并成一个巨大轮廓，
于是在该轮廓里 GT 的那几个空间**永远拿不到标签**；同时又对花园过度切分
（GT 一个 4.9 m² 的花园被切成 2.6 / 2.6 / 5.0 三块）。

**影响与对策**：

* 这不是 SAGE-E 的问题，也**对所有分类器一视同仁**（三个分类器吃的是同一批轮廓），
  所以比较仍然公平 —— 但它决定了任何分类器的**绝对上限**。
* 对策：**训练集一律用 RGP 轮廓导出**（`--contour-algo RGP`）。
  这也和 README 里"RGP 会切开 CDT 因未封口而合并的房间"的描述一致。
* 论文里值得写成一条方法论结论：**轮廓划分质量 = 空间类型分类的标签可达上限**，
  与分类器本身的强弱无关。
* ⚠️ 但这带来一个必须写清楚的边界：SAGE-E 的可预测类别集合由**所选轮廓算法**决定。
  报告数字时必须注明训练/推理用的是哪个轮廓算法。

### 5.4 阶段 E 结果（5-fold 交叉验证，按图纸切分）

**训练集**：`output/jsonld_rdp/`（RGP 轮廓）39 张图，8 维节点 / 5 维边特征。
标准化：逐列 z-score（每 fold 用该 fold 的训练集统计，避免泄漏）。
损失：按类频次反比的加权交叉熵。200 epoch，CPU 约 30 秒/fold。

| 指标 | **FULL**（18 类） | **CORE**（14 类） |
|---|---|---|
| Accuracy | 0.5007 ± 0.0530 | **0.5551 ± 0.0921** |
| Macro F1 | 0.2762 ± 0.0463 | **0.3457 ± 0.0790** |
| Weighted F1 | 0.4554 ± 0.0499 | **0.5273** |
| 合并后 Accuracy | 0.5004 | 0.5506 |
| 合并后 Macro F1 | — | 0.3875 |

> ⚠️ **别用单次划分的数字。** 同一份数据用 `train/valid/test = 27/4/8` 的单次划分
> 能得到 Accuracy 0.6018 —— 但那是**测试集偏容易**造成的乐观偏差（5 个 fold 的
> 单折准确率在 0.44–0.59 之间波动）。交叉验证的结果才是可信口径。

**CORE 空间更好的原因**（符合预期）：

1. 走廊三类合一 → `MainCorridor` 在训练集里从 7 个样本变成 `Corridor` 165 个；
2. 丢掉 `ElectricalRoom`(3)、`WaterRoom`(4) 这两个根本学不动的类。

**当前天花板在哪**：见 §5.3 —— 轮廓划分质量决定标签可达上限。
RGP 已经把 DiningRoom 从 3 → 39、SecondaryCorridor 从 9 → 62，但走廊细分
（主/次/公共）系统仍然分不出来，这是 macro-F1 上不去的主因。

### 5.5 集成与单图验证（阶段 E-2）

已接入 `ISpaceTypeClassifier`，注册名 `SAGEE`（别名 `SAGE` / `SAGE_E`），
Web UI 下拉可选。部署形态是**纯 numpy**：不需要 torch、不需要 DGL、不需要子进程。

用 `--contour-algo RGP --classifier-algo SAGEE` 跑通全流程：

```
[SAGEE] 35 个空间 -> 31 条预测（4 条因置信度 < 0.50 被丢弃）（35 房间 / 56 边，14 类）
🎨 Rendering graph: 35 spaces, 22 doors, 90 functional elements
```

逐空间与 GT 对比（`scripts/compare_space_types.py`，`2suite (1)`）：

| | |
|---|---|
| 完全一致 | **29 / 35 = 82.9%** |
| Bedroom | 6 / 6 ✅ |
| Garden | 6 / 6 ✅ |
| Balcony | 6 / 7 |
| LivingRoom / Kitchen / DiningRoom | 各 2 / 2 ✅ |
| ElevatorShaft | 1 / 1 ✅ |
| Corridor（CORE 合并后） | 4 / 5 |

⚠️ **两个必须注意的口径细节**：

1. **GT 标签必须先过与训练一致的折叠规则再比**。CORE 空间把
   `MainCorridor` / `SecondaryCorridor` / `PublicCorridor` 统一成了 `Corridor`；
   直接拿原始 GT 类型对比会把**正确的预测算成错误**（本例 71.4% → 82.9%）。
2. **6 个错误里有 4 个是「未判定」而非判错** —— 它们是置信度 < 0.5 被丢弃的**弃权**。
   也就是说 `min_confidence=0.5`（与其它分类器共享的默认值）对 SAGE-E 偏激进，
   值得在阶段 F 里作为超参扫描一下。

## 6. 评估口径（阶段 F）—— 三个必须遵守的规则

> ⚠️ **本节 6.1–6.6 的命令与数字均来自旧的 RGP 轮廓 + tuned 配方口径，已被 §6.7 取代。**
> 仍然成立的是这里的**规则本身**（不能样本内、弃权算 FN、标签先折叠）—— 规则一条都没变，
> 变的只是实验设置。要复现当前结果请直接看 **§6.7**。

### 6.1 ⛔ 绝不能用「样本内」数字

SAGE-E 是**需要训练**的分类器，而 `TextMatching` / `LLMMultiStage` 不需要。
这带来一个很容易踩的陷阱：

```
部署模型在全部 39 张图上训练
        ↓
又拿它在同一批 39 张图上推理 → 评估的其实是训练集
```

实测这个 in-sample 数字是 **0.717 acc / 0.746 macro-F1**，而真实的样本外是
**0.5506 / 0.3875** —— 虚高了 17 个点。

**识别方法**：看逐类样本分布是否与训练集完全一致
（Balcony 196 / Bedroom 353 / Corridor 165…）。一致就是在评估训练集。

**正确做法**：`scripts/run_holdout_eval.py` —— 5 折正好覆盖全部 39 张图，
让每张图都被**一个没见过它的模型**预测一次，拼起来就是完整的样本外**在线**结果，
与 TextMatching 的零样本结果口径一致。

```bash
# 1) 导出每折模型（torch 环境）
python scripts/train_sagee.py --dataset data/cad2graph_sagee_rgp_core.npz \
    --cv 5 --epochs 200 --normalize --class-weights --select final \
    --labels-from data/cad2graph_sagee_rgp_core.labels.json --emit-folds data/holdout_core

# 2) 留出评估（cadruler 环境）
python scripts/run_holdout_eval.py --folds data/holdout_core --thresholds 0.0 0.5
```

实测单折准确率：`0.4478 / 0.4533 / 0.5636 / 0.6413 / 0.6693`（每折 7–8 张图）。
**跨度 22 个百分点** —— 所以任何单次划分的数字都不可信。

### 6.2 ⛔ 弃权必须算漏检（FN）

第一版评估脚本把「未判定」列从 F1 的分母里去掉，结果 TextMatching 的
macro-F1 从 **0.462 被抬到 0.722（+56%）**。也就是说：

> **一个大量放弃判定的分类器反而显得更强。**

这足以让整个对比表的结论反过来。正确规则：

* `fn` 取自**整行**（含「未判定」与「空间外」两列）
* `fp` 只取自**真实类别列**（弃权/空间外不是一个"被预测的类别"，不产生 FP）

### 6.3 ⛔ GT 标签必须先折叠

见 §5.5。CORE 空间合并了走廊三类，直接比对会**把正确预测算成错误**
（71.4% vs 真实的 82.9%）。

### 6.4 ⚠️ `min_confidence` 必须扫描，不能用共享默认值

| 类别 | CV 里（强制 argmax、无弃权） | 在线（`min_confidence=0.5`） |
|---|---|---|
| Bathroom | F1 **0.547** | F1 **0.000**（145/149 弃权） |
| StudyRoom | F1 0.157 | F1 **0.000**（28/29 弃权） |

**模型完全能分辨卫生间**（CV F1 0.547），在线是 0 纯粹是因为 `0.5` 这个阈值
（三个分类器共享的默认值，对 SAGE-E 明显过激）把它们砍掉了。

为此加了 `SAGEE_MIN_CONFIDENCE` 环境变量，扫阈值不需要改配置文件。

### 6.5 ⚠️ 分类器会改变「空间集合」本身，分母因此不同

实测：**同一张图纸、同样的 RGP 轮廓、同样的构件数，不同分类器产出的空间数不一样。**

| 图纸 | SAGEE | TextMatching | 构件数 |
|---|---|---|---|
| `2suite (1)` | 35 | 35 | 112 / 112 |
| `2suite (3)` | 36 | **42** | 102 / 102 |
| `2suite (6)` | 29 | **35** | 86 / 86 |
| `2suite (7)` | 26 | **30** | 99 / 99 |

**原因**：富化过程里的 **ACD 复合空间切分**（见
`src/enricher/semantic_enricher.py::_apply_predictions` 里"专门为复合空间维护锚点列表，
供 ACD 模块切分时使用"）。

* `TextMatching` / `LLMMultiStage` 会给**一个空间打多个类型**（如 `Bedroom` +
  `LivingRoom`），被判定为复合空间 → 触发切分 → 空间变多、变小；
* `SAGEE` 是**单标签节点分类器**，一个节点只输出一个类型 → 永远不会被判为复合空间
  → 不会切分。

这不是实现缺陷，而是两类算法的**结构性差异**。后果：

1. 对比表里各分类器的**分母不同**，不能只看百分比；
2. 切分出来的碎片会被重复匹配到同一个 GT 空间，把该分类器的分母抬高；
3. 报告时必须显式标注这一点，或者改在**富化前**的空间集合上比较（后续工作）。

> 这也解释了为什么不能简单地把"可判空间数"当成同一个基数来做差值比较。

### 6.6 ★ 阶段 F 最终结果：按 **GT 空间**统计的在线对比

> ⚠️ **本节已被 §6.7 取代。** 这里的数字来自 **RGP 轮廓 + tuned 配方**（`--normalize` + `--class-weights`）
> 的旧口径；当前结论请看 §6.7。本节保留是因为它记录了两个仍然成立的发现：
> 两个分类器**几乎正交**，以及 `min_confidence` 必须取 0。

为了消掉 §6.5 的分母问题，`scripts/evaluate_per_gt_space.py` 改成**以 GT 空间为统计单位**：

1. 每个系统空间按几何对齐到一个 GT 空间（与训练集导出同一套 IoU / 质心逻辑）；
2. 落在同一 GT 空间下的多块系统空间**按多数票**合并成一个预测；
3. 在该 GT 空间上判对/判错 —— 不管它是被 1 块还是 3 块系统空间覆盖。

于是分母恒等于 GT 空间数，**切分与否都不影响可比性**。

**实验设置**：39 张图纸（5 折留出，每张图都由没见过它的模型预测）、1839 个 GT 空间、
CORE 14 类标签空间、RGP 轮廓、`--min-iou 0.3`。

| 分类器 | GT空间 | 准确率 | 覆盖率 | MacroF1 | WeightedF1 | 弃权数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SAGEE`（`min_confidence=0.0`） | 1839 | 0.378 | **0.680** | 0.317 | 0.424 | 589 |
| `SAGEE`（`min_confidence=0.5`） | 1839 | 0.321 | 0.461 | 0.305 | 0.411 | 992 |
| `TextMatching` | 1839 | 0.376 | 0.433 | 0.409 | 0.487 | 1042 |
| **`ENSEMBLE`（TextMatching → SAGEE 兜底）** | 1839 | **0.523** | **0.746** | **0.510** | **0.584** | 468 |

> ⚠️ `SAGEE` 的 589 个"弃权"主要不是分类器弃权（阈值 0 时它必答），而是
> **该 GT 空间根本没有系统轮廓与之对齐** —— 这是轮廓划分的覆盖问题，不是分类问题。
> `TextMatching` 的 1042 里则同时包含"没匹配上"和"没文本可读"两种。

**互补性（这是最有价值的一张表）**

| | GT 空间数 | 占比 |
| --- | ---: | ---: |
| 两者都判对 | 391 | 0.213 |
| **仅 SAGEE 判对** | **304** | **0.165** |
| **仅 TextMatching 判对** | **301** | **0.164** |
| 两者都判错 | 843 | 0.458 |
| 并集上界（任一判对） | 996 | 0.542 |
| 「TextMatching → SAGEE」规则组合器 | 962 | 0.523 |

**结论 1 —— 两个分类器几乎正交。** 各自"单独判对"的数量几乎相等（304 vs 301），
完全互不覆盖。SAGE-E 的价值不在于整体准确率更高，而在于**它答的是另一个子集**。

**结论 2 —— 一个简单规则就能吃掉几乎全部上限。** 「文本先判、判不出再问 SAGE-E」
拿到 962/1839 = 0.523，距离"并集上界"996 = 0.542 只差 **1.9 个百分点**，
即已经吃掉了可组合收益的 **96%**。加复杂的融合器收益很小。

**结论 3 —— SAGE-E 的贡献集中在"没有文字"的空间上。** 逐类 F1（节选）：

| GT 类别 | SAGEE | TextMatching | ENSEMBLE | 谁强 |
| --- | ---: | ---: | ---: | --- |
| `ElevatorShaft` | **0.615** | 0.000 | 0.632 | SAGEE（文本永远读不到电梯井） |
| `Stairwell` | **0.463** | 0.000 | 0.535 | SAGEE |
| `Bathroom` | **0.425** | 0.171 | 0.486 | SAGEE（卫生间常无文字标注） |
| `Corridor` | **0.358** | 0.309 | 0.492 | SAGEE（略胜） |
| `StudyRoom` | 0.122 | **0.833** | 0.735 | TextMatching |
| `Kitchen` | 0.398 | **0.772** | 0.738 | TextMatching |
| `Balcony` | 0.415 | **0.716** | 0.584 | TextMatching |
| `Cloakroom` | 0.105 | **0.634** | 0.531 | TextMatching |
| `DiningRoom` | 0.065 | **0.516** | 0.516 | TextMatching |

**结论 4 —— `min_confidence` 应当取 0。** 阈值 0.5 让准确率从 0.378 掉到 0.321、
覆盖率从 0.680 掉到 0.461，macro-F1 也略降。SAGE-E 的 softmax 系统性偏低
（实测中位数 0.966 但低分尾部很长），用阈值切只会砍掉正确答案。
**阶段 G 的阈值扫描至此已有结论，默认值应改为 0.0。**

**结论 5 —— 主要纠错空间在 `Balcony` 与 `Garden`。**
`Balcony` 是几乎所有类的最大误判去处（Bathroom→Balcony 35、Kitchen→Balcony 45、
Corridor→Balcony 45 …），典型的**类别不平衡导致的过度预测**；
`Garden` 两边 F1 都接近 0（SAGEE 0.043 / TextMatching 0.087），说明这个类靠现有特征
（面积、门数、拓扑）根本不可分。这两类是后续调优的第一优先级。

### 6.7 ★★ 当前口径：GT 轮廓 + 忠实配方的最终结果

§6.6 有两处已经不再成立，必须换成下面这一节：

| 旧口径 | 为什么错 | 现在 |
|---|---|---|
| 分类器各跑 **RGP 轮廓** | 差异里混进了轮廓误差，不是有效 ablation | 三种方法**统一在 GT 轮廓上** |
| 训练用 `--normalize --class-weights` | 论文里没有这两个东西 | **忠实配方**（两者都关，见 §4.2） |
| SAGEE 用 RGP 训练好的模型 | 喂 GT 轮廓是**分布外**评估，对 SAGEE 系统性不利（实测 0.607 / 0.357） | 用 GT 轮廓**重训** |

**实验设置**：39 张图纸、**1916 个 GT 空间**、`CORE` **16 类**、GT 轮廓、`--min-iou 0.3`、
SAGEE 为 5 折留出在线结果。权威产物：`output/bench/scores.json`。

| 分类器 | GT 空间 | 准确率 | 覆盖率 | MacroF1 | WeightedF1 | 弃权 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`LLMMultiStage`**（`deepseek-flash`） | 1916 | 0.677 | 0.825 | **0.637** | 0.731 | 336 |
| **`SAGEE`**（折外） | 1916 | **0.755** | **0.999** | 0.566 | **0.732** | 1 |
| `TextMatching` | 1916 | 0.449 | 0.469 | 0.429 | 0.532 | 1017 |

> `LLMMultiStage` 这一行用的是 **`deepseek-flash`**。换模型之前的 `glm-4.5-air`
> 跑出的是 `0.731 / 0.902 / 0.699 / 0.778`（旧产物归档在
> `output/bench/_archive_type_LLMMultiStage_glm/`）。**引用大模型数字时必须写明
> 模型名** —— 差异高达 5 个准确率点。SAGEE 与 TextMatching 不受影响。

**重训是必需的，否则结论是错的。** 未重训（RGP 模型 → GT 轮廓）时 SAGEE 只有
`0.607 / 0.357`，看上去远差于大模型；用 GT 轮廓重训后 `0.755 / 0.566`，
**准确率反超大模型**。

**标签空间从 14 类改成 16 类。** GT 轮廓上 `ElectricalRoom` / `WaterRoom` 真实存在
（RGP 会把它们合并掉，所以 RGP 版只有 14 类）。若沿旧 14 类，SAGEE 对这两类的预测
会被记成「空间外」，对它系统性不利。

**逐类 F1（绿色 = 最好）**

| GT 类别 | support | `LLMMultiStage` | `SAGEE` | `TextMatching` |
|---|---:|---:|---:|---:|
| Balcony | 272 | 0.818 | 0.762 | **0.818** |
| Bathroom | 222 | 0.602 | **0.823** | 0.336 |
| Bedroom | 362 | **0.940** | 0.850 | 0.724 |
| Cloakroom | 27 | **0.778** | 0.333 | 0.683 |
| Corridor | 236 | 0.572 | **0.841** | 0.008 |
| DiningRoom | 138 | 0.903 | 0.858 | **0.920** |
| ElectricalRoom | 48 | 0.480 | **0.550** | 0.000 |
| ElevatorShaft | 73 | 0.462 | **0.797** | 0.000 |
| Entrance | 52 | 0.404 | 0.230 | **0.457** |
| Garden | 66 | **0.409** | 0.396 | 0.114 |
| Kitchen | 144 | **0.864** | 0.638 | 0.819 |
| LivingRoom | 142 | 0.856 | 0.878 | **0.919** |
| Stairwell | 39 | 0.655 | **0.716** | 0.000 |
| StorageRoom | 35 | **0.458** | 0.150 | 0.000 |
| StudyRoom | 31 | **0.825** | 0.000 | **0.825** |
| WaterRoom | 29 | 0.170 | 0.230 | **0.242** |

**结论**（与 §6.6 的结论 1 一致，但分界更清楚）：

* **不存在全面胜者。** SAGEE 准确率高 **7.8 个点**且几乎不弃权（1/1916），
  但 MacroF1 低 **7.1 个点** —— 它在小类上崩得很厉害（`StudyRoom` **0.000**）。
  （换成 `deepseek-flash` 之前两者的差距更小：+2.7 / −12.8。）
* **分界线是"这个类型能不能靠几何/拓扑推出来"**：
  * 能 → SAGEE：`Corridor` 0.841（大模型 0.572）、`Bathroom` 0.823、`ElevatorShaft` 0.797、
    `Stairwell` 0.716（大模型 0.655）、`ElectricalRoom` 0.550；
  * 不能 → 大模型：`Kitchen` 0.864、`StudyRoom` 0.825（SAGEE **0.000**）、
    `Cloakroom` 0.778（SAGEE 0.333）、`StorageRoom` 0.458、`Bedroom` 0.940。
* `TextMatching` 在**所有"本来就没文字"的类别**上 F1 恒为 0
  （`Stairwell` / `ElevatorShaft` / `ElectricalRoom` / `StorageRoom`），这是它的结构上限，
  也解释了它 0.469 的覆盖率。

> ★ **端到端（任务 2 跑在任务 1 的产物上）的结果见 [benchmark_protocol.md](benchmark_protocol.md) §7.4。**
> 那里有一个本节看不到的发现：`RGP + SAGEE` 只比最优格低 1.9 个点且完全不需要大模型；
> 而 `CDT` 的过合并对 SAGEE 的伤害（条件准确率 0.637）远大于对大模型的（0.736），
> 因为 ACD 能把 CDT 合并出来的复合空间重新切开。

---

## 7. 接入设计（CAD2Graph 侧）

### 7.1 接口

`ISpaceTypeClassifier`（`src/spatial/contracts.py:59`）：

```python
def classify(self, *, contours=(), texts=(), components=(),
             adjacency=None, context=None) -> list[SpaceTypePrediction]
```

✅ 实测 `SemanticEnricher._build_contours()`（`src/enricher/semantic_enricher.py:100`）
已经喂好了我们需要的全部信息：

| `SpatialContour` 字段 | 内容 | 用途 |
|---|---|---|
| `id` / `geometry` | 轮廓 ID 与外环（mm） | 节点 + 几何类特征 |
| `area_sqm` | 面积 | f5 |
| `elements` | 落在轮廓内的构件 uid | 家具/门窗 → 节点特征 |
| `neighbors` | 由 `bot:adjacentZone` 填入 | **图的边** |
| `attributes['node']` | 原始图谱节点 | 兜底取任意字段 |

⚠️ `execute_enrichment()` **不传 `adjacency=`**，图靠 `contour.neighbors` 建立。

### 7.2 不依赖 DGL（重要）

模型只有 4 层线性 + 求和聚合，**完全不需要 DGL**：

```python
# 等价前向，仅 numpy / torch
m = np.zeros_like(h)
np.add.at(m, dst, np.maximum(h[src] @ W_msg[:, :8].T
                             + e     @ W_msg[:, 8:].T + b_msg, 0.0))
h = np.maximum(np.c_[h, m] @ W_apply.T + b_apply, 0.0)
```

好处：

* `cadruler` venv **不需要装 torch / DGL** —— 与 VecFloorSeg 的"子进程隔离"模式不同，
  这里可以**在进程内直接跑**；
* 训练可以放在 `vecfloorseg` conda 环境（已有 torch 2.1.2）或任意带 torch 的环境，
  训练完导出 `.npz`，推理侧只读 `.npz`；
* 免掉 DGL 那串依赖地狱（实测 DGL 2.2.1 需要 `torchdata` + `pydantic`，
  且 `torchdata` 版本必须匹配 torch，否则 `ModuleNotFoundError: torch.utils._import_utils`）。

### 7.3 模块划分

```
src/spatial/sagee/
├── __init__.py
├── features.py      # CAD2Graph 图 → 8/5 维特征（唯一的"设计"所在）
├── graph.py         # contours + neighbors → (node_feat, edge_index, edge_feat)
├── model.py         # 纯 numpy 前向 + 载入 .npz 权重（无 torch）
├── labels.py        # L0 本体映射表 + 长尾折叠规则
└── export.py        # 训练用：图 → npz 数据集（跑在带 torch 的环境）

src/spatial/classifiers/sagee.py     # ISpaceTypeClassifier 实现 + 注册
scripts/
├── export_roomgraph.py              # 解码 roomgraph.bin → npz（一次性）
├── train_sagee.py                   # 训练（torch，CPU 分钟级）
└── check_sagee_config.py            # 配置/映射表自检（仿 check_vecfloorseg_config.py）
```

### 7.4 配置

沿用 VecFloorSeg 已经跑通的模式：

```yaml
# settings.yaml
spatial:
  classification:
    algorithm: "LLMMultiStage"
    algorithm_params:
      SAGEEndToEnd:
        weights: ""          # 留空 -> 环境变量 SAGEE_WEIGHTS
        feature_schema: "cad2graph-v1"
        label_space: "CORE9" # CORE9 | FULL21
        use_text_features: false
```

---

## 8. 分阶段实施

| 阶段 | 内容 | 验收标准 | 预估 | 状态 |
|---|---|---|---|---|
| **A** | 端口 SAGE-E 前向（无 DGL）+ 解码 RoomGraph → npz | **在 RoomGraph 上复现 acc 0.797 / macro-F1 0.780** | 0.5 天 | ✅ 已完成（F1 四位小数一致、混淆矩阵 81 格全同） |
| **B** | `features.py` + `graph.py`：contours → 8/5 维图 | 单张图纸产出的图能被 A 的模型吃下；特征分布与 RoomGraph 逐列对比 | 1 天 | ⬜ |
| **C** | `labels.py`：L0 映射表 + 长尾折叠 + 单测 | 映射表通过合理性校验（Bedroom 数量级一致） | 0.5 天 | ⬜ |
| **D** | 训练集生成（复用 `src/experiment/evaluator.py` 已有的几何重叠映射） | 41 图 / 1951 样本落盘，按图纸 5-fold | 0.5 天 | ⬜ |
| **E** | 训练 + L1/L2 评估 + 接线（`settings.yaml` / Web UI / README） | 三个分类器在同一测试集上的对比表 | 0.5 天 | ⬜ |
| **F** | （可选，依赖 §4.3）合并 RoomGraph 做迁移实验 | `仅CAD2Graph` vs `预训练+微调` 对比 | 0.5 天 | ⬜ |

**A 阶段是硬门槛** —— ✅ 已通过。产出：

| 文件 | 作用 |
|---|---|
| `third_party/SAGE-E/` | submodule（MIT，含 4 个缺失的官方资产） |
| `scripts/export_roomgraph.py` | `roomgraph.bin` → `data/roomgraph.npz`（一次性的，只在导出时需要 DGL） |
| `src/spatial/sagee/dataset.py` | 扁平 npz 数据集 + 划分 + 指标（**只依赖 numpy**） |
| `src/spatial/sagee/model.py` | SAGE-E 的 numpy 前向（**无 torch / DGL**） |
| `scripts/train_sagee.py` | torch 训练 / 评测 / 导出 `.npz` 权重 |
| `scripts/check_sagee_port.py` | 回归测试：numpy 端口 vs torch 基准逐位对比 |

⚠️ **环境下坑（已解决）**：`src/spatial/__init__.py` 会 eager import 所有轮廓算法，
而 `rgp.py` 用了 PEP 585 的运行期下标（`tuple[float, float]`），**需要 Python 3.9+**；
但训练环境被 VecFloorSeg 的 torch 1.13 锁死在 **Python 3.8**。
因此 `train_sagee.py` 用 `importlib` 按路径直接加载 `sagee/{dataset,model}.py`
（这两个模块只依赖 numpy），绕开包机制，**不需要动训练环境的 Python 版本**。

---

## 9. 风险

| # | 风险 | 严重度 | 对策 |
|---|---|---|---|
| 1 | 8/5 维特征语义未公开 | 🔴 高 | 三条确认路径见 §4.3；**不阻塞**自有数据训练 |
| 2 | 9 类名称/顺序未公开 | 🟡 中 | 用"每图平均个数"做经验校验（§3.3 已测）；必要时问作者 |
| 3 | 类别体系 9 vs 21 | 🟡 中 | L1/L2 分层评估（§5.2） |
| 4 | 图数量只有 41 | 🟡 中 | 按图纸 5-fold CV；⚠️ `{2,3,4,6}suite (1..10)` 可能是同族变体，需检查近重复泄漏 |
| 5 | 长尾类样本极少 | 🟡 中 | 合并规则 + 同时报 macro/weighted/top-3/混淆矩阵 |
| 6 | 迁移学习可能负收益 | 🟢 低 | 实测对比，负收益就如实报告 |
| 7 | 训练/推理环境的 torch 依赖 | 🟢 低 | 推理走 numpy，只在训练时用 torch |

---

## 10. 复现清单

> 当前口径（GT 轮廓 + 忠实配方）的命令见 **§6.7**；两任务基准见
> [benchmark_protocol.md](benchmark_protocol.md) §5。下面是端口复现与自检。

```bash
# 一次性：解码 RoomGraph（需要 dgl + torchdata + pydantic，建议放独立目录）
python scripts/export_roomgraph.py --out D:\Dev\BIM\_sagee\roomgraph.npz

# A 阶段：端口自检 —— 必须复现 best_user.pt 的 acc 0.7928 / F1 0.7801（见 §3.5）
# 本命令在 cadruler 环境即可跑（无 torch、无 DGL）
python scripts/check_sagee_port.py

# 当前训练集：从 GT 轮廓的 run 目录导出
python scripts/export_sagee_dataset.py --jsonld-dir output/bench/type_SAGEE \
    --gt-dir output/gt --space CORE --out data/cad2graph_sagee_gtc

# 配置 / 三件套自检
python scripts/check_sagee_config.py
```

**已下载的素材**（在 `D:\Dev\BIM\_sagee\`，未进仓库）：
`roomgraph.bin`、`best_default.pt`、`best_user.pt`、`SAGEE.py`、
`w78-2021-paper-077.pdf` / `.txt`、`lib/`（dgl + torchdata + pydantic + pypdf 的临时安装）。

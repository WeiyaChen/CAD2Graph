# 基准评测协议（两个任务 · 可复现的 ablation）

> 本文档定义 CAD2Graph 的**可插拔空间算法**如何被公平地比较，以及当前的实测结果。
> 它取代了早期散落在 README 里的口径说明 —— 那些数字来自不同时点、不同图纸集合、
> 不同轮廓算法，**彼此不可比**，不要再引用。
>
> 所有数字都来自 `output/bench/scores.json`，由 `scripts/evaluate_benchmark.py` 生成。
> 截图式 UI 见 `exp_analysis.html` → ③ 两项任务综合比较。

---

## 1. 为什么要重做评测

早期的对比表有三个致命问题，任意一个都足以让结论反过来：

| 问题 | 后果 | 现在的做法 |
|---|---|---|
| 每次实验新开一个输出目录，口径/图纸集合/算法组合各不相同 | 分母不同，差值无意义 | 一次运行 = 一个 `output/bench/<run>/` 目录，目录名即口径 |
| 分类器在**不同轮廓**上评估 | 差异里混进了轮廓误差 | 任务 2 三种方法**全部跑在 GT 轮廓上** |
| SAGE-E 用样本内数字 | 虚高 17 个百分点 | 5 折留出，每张图都由**没见过它**的模型预测一次 |

---

## 2. 两个任务，不做交叉

这是本协议的核心。轮廓提取和类型识别是**两个独立任务**，产物互不共用：

```
                     ┌──────────────────────────────────────────┐
                     │  input_data/svg/<drawing>.svg            │
                     └────────────────┬─────────────────────────┘
                                      │
             ┌────────────────────────┴────────────────────────┐
             │                                                 │
    ┌────────▼─────────┐                            ┌──────────▼──────────┐
    │ 任务 1 · 轮廓提取 │                            │ 任务 2 · 类型识别    │
    │ contour_*        │                            │ type_*              │
    │ 分类器 = NoOp    │                            │ 轮廓 = GT           │
    └────────┬─────────┘                            └──────────┬──────────┘
             │                                                 │
   ┌─────────┴──────────┐                          ┌───────────┴────────────┐
   │ CDT / RGP /        │                          │ LLMMultiStage /        │
   │ VecFloorSeg        │                          │ SAGEE / TextMatching   │
   └─────────┬──────────┘                          └───────────┬────────────┘
             │                                                 │
   ┌─────────▼──────────┐                          ┌───────────▼────────────┐
   │ 读 <base>_raw.jsonld│                          │ 1916 个 GT 空间        │
   │ 量轮廓几何          │                          │ 量分类正确性           │
   └────────────────────┘                          └────────────────────────┘
```

### 2.1 任务 1 为什么必须用 `NoOp` 分类器

1. **流水线必须有分类器。** 若落到 `settings.yaml` 的默认值（`LLMMultiStage`），
   一次纯轮廓实验会白烧 39+ 次大模型调用。
2. **分类器会改变轮廓本身。** 富化阶段的 **ACD 复合空间切分**会把"被判了多个类型"
   的空间切开 —— 实测同一张图在 `TextMatching` 下 42 个空间、单标签分类器下 36 个。
   拿带分类器的产物去量轮廓，量到的是"分类器 + 提取器"的混合物。

**评估读的也是 `<base>_raw.jsonld`**（富化**之前**写出的那份图，只有 `bot:Space`
和几何）。这才是"提取出来的轮廓"本身。

> **"与分类器无关"的精确含义（已实测）**：同一张图在 `type_SAGEE` /
> `type_LLMMultiStage` / `type_TextMatching` 三个目录下，`_raw.jsonld` 的文件大小、
> `@id` 集合、空间数、WKT 几何、`bot:containsElement` 的内容**全部一致**；
> 但**字节不全等** —— 差在 `bot:adjacentZone` / `bot:containsElement` 这类
> **无序关系列表的元素顺序**（与 README「Notes」里记录的非确定性同源）。
> 所以**不要按字节比对**，要比就比集合。

### 2.2 任务 2 为什么必须统一在 GT 轮廓上

若三种分类器各跑各的轮廓，差异里就混进了轮廓误差。用 GT 轮廓当公共起点，
测到的才是**分类能力本身**。另外两种（`TextMatching` / `LLMMultiStage`）天然零样本；
`SAGEE` 用**折外**模型（见 §4）。

> `GT` 轮廓提取器（`src/spatial/contours/gt.py`）就是为此新增的：它读
> `output/gt/<base>_gt.jsonld` 的 WKT 多边形，**把所有空间的类型都标成 `Unknown`**，
> 绝不泄漏 GT 标签给分类器。它的 `visualization_suffix` 是 `gt`。

### 2.3 端到端：两者的排列组合（`e2e_*`）

上面 §2.1 与 §2.2 是**算法级**的局部对比 —— 它们故意把两个任务隔开，好回答
"这个轮廓算法好不好"、"这个分类器好不好"。

但真实使用是把两个任务**串起来**：先提取轮廓，再在**提取出来的轮廓上**判类型。
误差会叠加，而且叠加的方式不一定可加 —— 轮廓漏掉一个空间，分类器再准也白搭。
所以还需要一组**端到端**的 run：

```
3 种轮廓算法  ×  3 种分类器  =  9 个 run   （e2e_<轮廓算法>_<分类器>）
```

与 §2.2 的关键差别：**任务 2 不再跑在 GT 轮廓上**，而是直接用任务 1 的产物。
度量方式不变（仍按 GT 空间统计、弃权算漏检、标签先折叠），但每格的分母里
包含了轮廓误差。

评估时把准确率拆开，才知道损失到底出在哪一环：

```
端到端准确率  =  轮廓覆盖  ×  条件准确率
                (任务1 找出该 GT 空间了吗) × (找出之后类型判对了吗)
```

* **轮廓覆盖** = `1 − (混淆矩阵最后一列 <空间外> 的占比)` —— 没找到就直接判错，与分类器无关；
* **条件准确率** = 判对数 / 已覆盖数。

于是可以直接读出结论：「端到端只有 0.62，但轮廓覆盖已经 0.94，所以主要是分类的锅」
或者相反。这是这个矩阵唯一真正有价值的地方 —— 否则九个数摆在一起并不能告诉你该优化什么。

⚠️ **端到端与算法级的数字不可直接相减当作"轮廓的代价"。** 因为任务 2 在
不同轮廓上产生的空间集合不同，ACD 切分行为也不同（见 §4.1）。分解表给出的是
诊断信息，不是严格的误差传播分析。

---

## 3. 基准集 = 39 张

`input_data/dxf/` 下共 41 个图纸，排除两个：

| 图纸 | 排除原因 |
|---|---|
| `sample` | 只是跑通流程的样例，用户明确要求不计入 |
| `6suite (5)` | 它的 GT 文件是空的（`{"@graph": []}`），没有任何可评的东西 |

键的规范化规则（`src/experiment/bench_metrics.py::drawing_key`）：

```
2suite_annotated (3).dxf  →  2suite#3
```

`--mode BATCH` 无法按文件名过滤，所以运行阶段仍会跑 `sample`，**统一在评估阶段排除**
（`--exclude sample`，两个脚本都要传，且必须一致）。

---

## 4. 公平性原则（写进代码的三条）

这三条不是文档约定，而是**代码里强制的**，因为违反任意一条都会静默地翻转结论：

### 4.1 ACD 只给 `LLMMultiStage` —— 归属于创新点，不是通用算子

ACD（复合空间切分/聚合）是**本方法提出的创新**，SAGE-E 原文没有这个能力。
若把它同时挂给 `TextMatching`，等于给基线白送增益，对比就不再是"方法 vs 方法"。

实现在 `ISpaceTypeClassifier.supports_composite_split`（`src/spatial/contracts.py`）：

```python
class ISpaceTypeClassifier:
    #: 是否支持对复合空间做"切分/聚合"。
    #: 默认 False —— 只有本方法的分类器才允许打开，否则就是给基线白送增益。
    supports_composite_split: bool = False
```

* `LLMMultiStage` → `True`（唯一一个）
* `TextMatching` / `SAGEE` / `NoOp` → 继承默认 `False`

`src/enricher/enricher_pipeline.py` 读取该标志，为 `False` 时跳过 ACD 并打印说明。

**A/B 实测**（同 RGP 轮廓 + `TextMatching`，4 张图）：

| | 空间数 |
|---|---:|
| ACD 关闭 | 126 |
| ACD 开启 | 142（多出 16 个碎片） |

### 4.2 忠实复现 —— 不加论文里没有的前后处理

SAGE-E 原文的配置是 **不标准化、不加类别权重**。早期版本自作主张加了 z-score 与
class-weights，理由是"量纲差异大不标准化训不动"—— **这个判断是错的**，实测忠实配置
反而更好：

| 配方 | 准确率 | MacroF1 | WeightedF1 |
|---|---:|---:|---:|
| `faithful`（论文原样） | **0.6004 ± 0.0366** | 0.3334 | **0.5525** |
| `tuned`（+z-score +class-weights） | 0.5551 ± 0.0921 | **0.3457** | 0.5273 |

现在两个开关都默认 **关闭**，且训练时会把配方写进 `<weights>.meta.json`：

```json
{"recipe": "faithful", "normalize": false, "class_weights": false,
 "epochs": 200, "lr": 0.01, "weight_decay": 0.0005, "dropout": 0.5,
 "seed": 42, "n_class": 16, "dim_node": 8, "dim_edge": 5}
```

**⚠️ 一个曾经的静默陷阱**：`SageeNumpy` 会自动加载同名的 `.norm.json` 边车文件。
切回忠实配方后这个孤儿文件还在，于是被套用到一个**从没用过它的模型**上。
现在 `src/spatial/classifiers/sagee.py::_resolve()` 会读 `.meta.json`：

* `meta['normalize'] is True` 但文件缺失 → **报错**；
* `meta['normalize'] is False` → **忽略**已存在的 norm 文件；
* 标签文件始终必需。

`scripts/train_sagee.py::drop_stale_norm()` 在导出时会清掉过期的 norm 文件。

### 4.3 弃权算漏检

第一版评估把「未判定」列从 F1 分母里去掉，TextMatching 的 macro-F1 从 **0.462 被抬到
0.722（+56%）** —— 一个大量放弃判定的分类器反而显得更强。

正确规则（`src/experiment/bench_metrics.py::metrics_from_cm`）：

* `fn` 取自**整行**（含「未判定」「空间外」两列）
* `fp` 只取自**真实类别列**（弃权不是一个"被预测的类别"，不产生 FP）
* `accuracy` 永远与 `coverage` 一起报，让权衡可见

### 4.4 GT 标签必须先折叠

GT 里走廊被细分成 `MainCorridor` / `SecondaryCorridor` / `PublicCorridor` 三类，
而方法侧只输出合并后的 `Corridor`。不折叠就会**把正确预测算成错误**（71.4% vs 真实 82.9%）。

折叠表随 `scores.json` 一起发布（`label_space.folding`），UI 也用同一份表：

```json
{"MainCorridor": "Corridor", "SecondaryCorridor": "Corridor",
 "PublicCorridor": "Corridor", "SunRoom": "Balcony",
 "VentilationRoom": "ElectricalRoom"}
```

---

## 5. 怎么跑

```bash
# 0) 看当前进度（哪些 run 已完整）
python scripts/run_benchmark.py --status

# 1) 跑基准矩阵（只跑不完整的；3 + 3 = 6 个 run，每个 39 张图）
python scripts/run_benchmark.py --dry-run          # 先看要跑什么
python scripts/run_benchmark.py                    # 全跑
python scripts/run_benchmark.py --task contour     # 只跑任务 1
python scripts/run_benchmark.py --run contour_RGP --force

# 2) 评估（生成 output/bench/scores.json）
python scripts/evaluate_benchmark.py --out-json output/bench/scores.json
```

两个脚本的 `--exclude` 默认都是 `sample`，**必须保持一致**，否则评估会去要一个
没跑过的图纸产物。

⚠️ **LLM 调用的成本控制。** `--mode BATCH` 无法按文件名过滤，所以直接跑会连
`sample` 一起解析。如果不想为它烧一次大模型调用，可以先做一个只含基准集的输入目录
（`output/bench_input_svg/`，39 个硬链接），再传给 `--target-dir`：

```bash
python scripts/run_benchmark.py --target-dir D:\Dev\BIM\CAD2Graph\output\bench_input_svg
```

⚠️ `--target-dir` 在流水线里是**相对 `input_data/svg/` 解析**的，所以上面必须给
**绝对路径**，给个目录名会变成 `input_data/svg/<名字>` 而找不到。

### 运行矩阵

| run 目录 | 轮廓算法 | 分类器 | 任务 |
|---|---|---|---|
| `contour_CDT` | `CDT` | `NoOp` | 1 |
| `contour_RGP` | `RGP` | `NoOp` | 1 |
| `contour_VecFloorSeg` | `VecFloorSeg` | `NoOp` | 1 |
| `type_LLMMultiStage` | `GT` | `LLMMultiStage` | 2 |
| `type_SAGEE` | `GT` | `SAGEE` | 2 |
| `type_TextMatching` | `GT` | `TextMatching` | 2 |
| `e2e_CDT_LLMMultiStage` | `CDT` | `LLMMultiStage` | 端到端 |
| `e2e_CDT_SAGEE` | `CDT` | `SAGEE` | 端到端 |
| `e2e_CDT_TextMatching` | `CDT` | `TextMatching` | 端到端 |
| `e2e_RGP_LLMMultiStage` | `RGP` | `LLMMultiStage` | 端到端 |
| `e2e_RGP_SAGEE` | `RGP` | `SAGEE` | 端到端 |
| `e2e_RGP_TextMatching` | `RGP` | `TextMatching` | 端到端 |
| `e2e_VecFloorSeg_LLMMultiStage` | `VecFloorSeg` | `LLMMultiStage` | 端到端 |
| `e2e_VecFloorSeg_SAGEE` | `VecFloorSeg` | `SAGEE` | 端到端 |
| `e2e_VecFloorSeg_TextMatching` | `VecFloorSeg` | `TextMatching` | 端到端 |

只跑某一组：

```bash
python scripts/run_benchmark.py --task e2e                 # 只跑9 个端到端 run
python scripts/run_benchmark.py --run e2e_RGP_SAGEE        # 只跑一个
```

外加一个**不参评**的 `GT(自检)` 行：拿 GT 轮廓对 GT 自己。它的
覆盖率/一对一率/mIoU 应当 ≈ 1.0 —— 不是 1.0 就说明匹配或指标实现有 bug。

```
GT(自检)   n_sys 1916  coverage 0.9995  one_to_one 0.9990  mIoU 1.0000
```

---

## 6. 指标定义

### 任务 1（`bench_metrics.contour_metrics`）

先把系统空间与 GT 空间做匹配：**IoU ≥ `min_iou`（默认 0.3）优先，否则退化为
"系统空间质心落在 GT 多边形内"**。

| 指标 | 定义 | 说明 |
|---|---|---|
| `n_sys` / `n_gt` | 系统 / GT 空间总数 | |
| `count_ratio` | `n_sys / n_gt` | 1.0 = 数量对了；< 1 漏、> 1 碎 |
| `coverage` | `n_covered_gt / n_gt` | GT 空间被任何系统空间盖住的比例 |
| `one_to_one` | 一对一匹配的 GT 空间比例 | 惩罚"多块碎片拼一个房间" |
| `miou_matched` | 在匹配上的对上求平均 IoU | **形状精度** |
| `area_mae_m2` | 匹配对的面积绝对误差均值 | 单位 m² |

### 任务 2（`bench_metrics.metrics_from_cm`）

以 **GT 空间**为单位：每个系统空间对齐到一个 GT 空间，落在同一 GT 空间下的多块
**按多数票合并成一个预测**，该 GT 空间只被计分一次。于是分母恒等于 GT 空间数，
切分与否都不影响可比性。

混淆矩阵是 `(n_class, n_class + 2)`：前 `n_class` 列是真实类别，第 +1 列是
`<未判定>`（分类器弃权），第 +2 列是 `<空间外>`（该 GT 空间根本没有系统轮廓与之对齐）。

| 指标 | 定义 |
|---|---|
| `accuracy` | 判对 / GT 空间总数（**含弃权与空间外，二者都算错**） |
| `coverage` | 有实际输出的 GT 空间比例 |
| `macro_f1` | 逐类 F1 的算术平均（**弃权列不进 F1 分母，但进 `fn`**） |
| `weighted_f1` | 按 `support` 加权的 F1 |
| `f1_by_class` / `support` | 逐类 F1 与样本数 |

---

## 7. 实测结果（39 张图）

### 7.1 任务 1 · 空间轮廓提取

| 轮廓算法 | 系统数 | 数量比 | 覆盖率 | 一对一率 | mIoU | 面积 MAE |
|---|---:|---:|---:|---:|---:|---:|
| `CDT` | 1205 | 0.629 | 0.547 | 0.483 | **0.860** | **3.30 m²** |
| `RGP` | 1606 | **0.838** | **0.656** | **0.545** | 0.754 | 3.47 m² |
| `VecFloorSeg` | 65 | 0.034 | 0.013 | 0.000 | 0.074 | 183.04 m² |
| `GT(自检)` | 1916 | 1.000 | 0.999 | 0.999 | 1.000 | 0.00 m² |

> GT 空间总数：1916。

**CDT 与 RGP 是一个权衡，没有全面胜者**：

* CDT 的 mIoU 高 **10.6 个点** —— 它**提取出来的形状更准**；
* RGP 的覆盖率（0.656 vs 0.547，**+10.9 个点**）和一对一率（0.545 vs 0.483，
  **+6.2 个点**）都更高 —— 它**能找到更多房间**，且更少出现"多块碎片拼一个房间"。
* 原因见 README 的 RGP 第 9 步：RGP 是**过滤**而不是**吸收**，小于 `min_area_mm2`
  的碎块（墙腔、楼梯/管道格）被丢弃而不是并入邻居，所以数量偏高而总面积略少。

选哪个取决于下游更怕"漏空间"还是更怕"形状不准"。

> ⚠️ **`VecFloorSeg` 的数字是坏的，但仍然保留。** `best1.ckpt` 的 `best` 是
> **epoch 1** —— 权重基本没训好，1291 个 region 收敛成 1 条轮廓
> （标签分布 `0=1083, 3=2, 11=206`，房间类命中率 16.1%）。用本地 `0.ckpt` 做对照
> （`0=1126, 2=165`，命中率 0.0%，0 条轮廓）证明**管线是通的，是权重欠训练**。
> 这是上游模型的问题，不是适配器的 bug。

### 7.2 任务 2 · 空间类型识别（统一在 GT 轮廓上）

| 分类器 | GT 空间 | 准确率 | 覆盖率 | MacroF1 | WeightedF1 | 弃权 |
|---|---:|---:|---:|---:|---:|---:|
| **`LLMMultiStage`** | 1916 | 0.731 | 0.902 | **0.699** | **0.778** | 187 |
| **`SAGEE`**（折外） | 1916 | **0.755** | **0.999** | 0.566 | 0.732 | 1 |
| `TextMatching` | 1916 | 0.449 | 0.469 | 0.429 | 0.532 | 1017 |

**不存在"最好的分类器"，三者回答的是不同问题：**

* `SAGEE` **准确率最高**（0.755）且几乎不弃权（1/1916）—— 它对每个空间都给答案。
  但 MacroF1 最低，因为它在小类上崩得厉害（`StudyRoom` **0.000**）。
* `LLMMultiStage` MacroF1 最高（0.699）—— 它**在小类上明显更强**
  （`Cloakroom` 0.769 vs 0.333、`Stairwell` 0.947 vs 0.716、`Garden` 0.810 vs 0.396）。
* `TextMatching` 覆盖率只有 0.469 —— **一半的空间它根本没话可说**，且在所有
  "本来就没有文字"的类别上 F1 恒为 0（`Stairwell` / `ElevatorShaft` /
  `ElectricalRoom` / `StorageRoom` 全是 0.000）。

### 7.3 逐类 F1

绿色 = 该类别最好（`exp_analysis.html` 里也是这么标的）。

| GT 类别 | support | `LLMMultiStage` | `SAGEE` | `TextMatching` |
|---|---:|---:|---:|---:|
| Balcony | 272 | 0.830 | 0.762 | **0.818** |
| Bathroom | 222 | 0.777 | **0.823** | 0.336 |
| Bedroom | 362 | **0.932** | 0.850 | 0.724 |
| Cloakroom | 27 | **0.769** | 0.333 | 0.683 |
| Corridor | 236 | 0.568 | **0.841** | 0.008 |
| DiningRoom | 138 | 0.900 | 0.858 | **0.920** |
| ElectricalRoom | 48 | 0.462 | **0.550** | 0.000 |
| ElevatorShaft | 73 | 0.777 | **0.797** | 0.000 |
| Entrance | 52 | 0.386 | 0.230 | **0.457** |
| Garden | 66 | **0.810** | 0.396 | 0.114 |
| Kitchen | 144 | **0.832** | 0.638 | 0.819 |
| LivingRoom | 142 | 0.882 | 0.878 | **0.919** |
| Stairwell | 39 | **0.947** | 0.716 | 0.000 |
| StorageRoom | 35 | **0.249** | 0.150 | 0.000 |
| StudyRoom | 31 | **0.825** | 0.000 | **0.825** |
| WaterRoom | 29 | 0.242 | 0.230 | **0.242** |

**读法**：挑方法时优先看你在意的类别，而不是只看总平均。

* 几何/拓扑能推出来的（`Bathroom` / `Corridor` / `ElectricalRoom` / `ElevatorShaft`）
  → SAGE-E 强；
* 需要常识或文字小类的（`Cloakroom` / `Garden` / `Kitchen` / `Stairwell` / `StudyRoom`）
  → 大模型强；
* `TextMatching` 的 0.000 全部落在**没有文字标注**的类别上，这是它的结构上限。

---

## 8. SAGE-E 的折外训练

**绝不能引用样本内数字。** 识别方法：看逐类样本分布是否与训练集完全一致
（Balcony 196 / Bedroom 353 / Corridor 165…）—— 一致就是在评估训练集。实测这个
in-sample 数字是 **0.717 acc / 0.746 macro-F1**，真实的样本外是 **0.5506 / 0.3875**，
**虚高 17 个点**。

正确做法是 5 折留出：5 折正好覆盖全部 39 张图，每张图都被一个没见过它的模型预测一次，
拼起来就是完整的样本外**在线**结果，与另外两个零样本分类器口径一致。

```bash
# 1) 从 GT 轮廓导出训练集（cadruler 环境）
#    ⚠️ export_sagee_dataset.py 没有 --contour-algo，它只读 --jsonld-dir 下的
#    <name>_raw.jsonld。所以要借一个 --contour-algo GT 的 run 目录当数据源；
#    用哪个都行，因为 _raw.jsonld 写在富化之前，与分类器无关（见 §2.1）。
python scripts/export_sagee_dataset.py --jsonld-dir output/bench/type_SAGEE \
    --gt-dir output/gt --space CORE --out data/cad2graph_sagee_gtc

# 2) 训练 + 导出每折模型（torch 环境 = vecfloorseg conda env）
#    --normalize / --class-weights 都不要加（见 §4.2）
python scripts/train_sagee.py --dataset data/cad2graph_sagee_gtc.npz --cv 5 \
    --epochs 200 --select final \
    --labels-from data/cad2graph_sagee_gtc.labels.json \
    --emit-folds data/holdout_gtc

# 3) 折外评估 → 直接写进 bench 目录（cadruler 环境）
python scripts/run_holdout_eval.py --folds data/holdout_gtc --thresholds 0.0 \
    --out-root output/bench --out-name type_SAGEE --contour-algo GT
```

⚠️ **训练与推理必须使用同一个轮廓算法。** 旧版部署模型是在 **RGP** 轮廓上训练的，
拿它去跑 GT 轮廓就是**分布外**评估（实测只有 0.607 / 0.357，不公平）。
当前模型训练于 **GT 轮廓**，这就是任务 2 能用 GT 轮廓做公共起点的前提。

⚠️ **`min_confidence` 必须取 0.0。** SAGE-E 的 softmax 系统性偏低，阈值 0.5 只会
砍掉正确答案（旧口径下准确率 0.378 → 0.321、覆盖率 0.680 → 0.461）。

---

## 9. 环境依赖（两个 Python 环境）

| 步骤 | 环境 | Python | 原因 |
|---|---|---|---|
| 跑基准矩阵 / 评估 | `cadruler` venv | 3.13 | 主环境，有 numpy + shapely |
| 训练 SAGE-E / VecFloorSeg 推理 | `vecfloorseg` conda | 3.8 | 需要 torch |

```text
cadruler     D:\Dev\BIM\CAD2Graph\cadruler\Scripts\python.exe
vecfloorseg  C:\Users\admin\miniconda3\envs\vecfloorseg\python.exe
```

⚠️ Python 3.8 **无法 import `src.spatial`**（`rgp.py` 用了 PEP 585 的运行时下标）。
`scripts/train_sagee.py` 因此用 importlib 按文件路径加载模块，绕过包导入。

---

## 10. 新增一个算法要改什么

1. 在 `src/spatial/contours/` 或 `src/spatial/classifiers/` 里实现接口并注册；
2. 在 `src/spatial/*/__init__.py` 里 import，让注册生效；
3. 在 `scripts/run_benchmark.py` 的 `RUNS` 里加一行：

```python
RUNS = [
    ('contour_CDT', 'CDT', 'NoOp', ('contour',)),
    ...
    ('contour_MyAlgo', 'MyAlgo', 'NoOp', ('contour',)),          # ← 新增任务1
    ('e2e_MyAlgo_LLMMultiStage', 'MyAlgo', 'LLMMultiStage', ('e2e',)),  # ← 新增端到端
]
```

4. 跑 `python scripts/run_benchmark.py --run contour_MyAlgo`，再跑评估。
   `evaluate_benchmark.py` 是按**目录名前缀**（`contour_` / `type_` / `e2e_`）自动发现的，
   不需要改评估代码。端到端的目录名必须是 `e2e_<轮廓算法>_<分类器>`
   （假定两个算法名里都不含下划线）。

---

## 11. 产物清单

```text
output/bench/
├── manifest.json                 # run 矩阵的执行记录
├── scores.json                   # ★ 权威结果（含 gt_paths、label_space.folding、e2e_task）
├── contour_CDT/                  # 任务 1：每个 run 一个目录，内含 39 张图 × 2 个产物
├── contour_RGP/
├── contour_VecFloorSeg/
├── type_LLMMultiStage/           # 任务 2：起点是 GT 轮廓
├── type_SAGEE/
├── type_TextMatching/
└── e2e_<轮廓算法>_<分类器>/       # 端到端：任务 2 跑在任务 1 的轮廓上（9 个）
```

每个 run 目录里实际只有两种文件（`<base>` = 去掉扩展名的图纸名）：

| 文件 | 内容 |
|---|---|
| `<base>.jsonld` | 富化后的最终 JSON-LD —— **任务 2** 的评估读这一份 |
| `<base>_raw.jsonld` | **富化前**的图 —— **任务 1** 的评估读这一份 |

`<base>_raw.jsonld` 与 `<base>.jsonld` **是两次独立写盘**，所以不要用 `_raw.jsonld`
去验证任务 2、也不要用 `.jsonld` 去量轮廓。

轮廓可视化 **不在** run 目录里，统一在 `output/viz/` 下，文件名后缀由算法的
`visualization_suffix` 决定：

| 算法 | 后缀 | 路径 |
|---|---|---|
| `CDT` | `cdt` | `output/viz/<base>_cdt.png` |
| `RGP` | `rgp` | `output/viz/<base>_rgp.png` |
| `VecFloorSeg` | `vecfloorseg` | `output/viz/<base>_vecfloorseg.png` |
| `GT` | `gt` | `output/viz/<base>_gt.png` |

---

## 12. 相关文档

* [docs/sagee_baseline_plan.md](sagee_baseline_plan.md) — SAGE-E 端口、数据契约、逐条实测
* [docs/vecfloorseg_baseline_plan.md](vecfloorseg_baseline_plan.md) — VecFloorSeg 适配器
* [docs/vecfloorseg_adapter_design.md](vecfloorseg_adapter_design.md) — 适配器 I/O 契约与上游坑
* [README.md](../README.md) — 环境、脚本、Web UI

# VecFloorSeg 适配层设计（CAD2Graph → 房间轮廓）

> 状态：**已全部实现并已进入基准矩阵**（阶段 A–D 完成；阶段 E 拿到了实测数字，结论是
> "权重欠训练"而不是适配层有问题，见 §5 与 §6.5）。
> 本文档记录的是**经源码逐行验证**的事实，不是推测。
> 相关：`docs/vecfloorseg_baseline_plan.md`（环境/训练方案 §7–§11）、`scripts/setup_vecfloorseg_server.sh`。

---

## 1. 结论先行：为什么不能走他们原来的 SVG 路线

作者的预处理链是 `model.svg → SVGParserCUBI → delaunayTriangulation → 图构建 → pkl`。
但：

1. `Replace_with_CubiCasa/roughcast_data_generation.py` **不是生成器**，只是一个 minidom 过滤器——它从 CubiCasa-5k 的 `model.svg` 里**保留** `<g id="Wall|Railing|Door|Window">` 等分组，几何完全来自 CubiCasa。
2. 真正决定输入的是 `SVGParserCUBI.getWallShape()`，它只认 CubiCasa 的 `<g id=…>` + `<polygon>` 约定，**不支持任何曲线/弧**（`NotImplementedError`）。
3. 更要紧的是：`delaunayTriangulation` 真正需要的只是一个 **PSLG**（`verticesDict = {(x,y): [vIdx, nbr…]}`），**而 CAD2Graph 的 `walls` 本身就是这个 PSLG**。

→ **我们跳过 SVG，直接把 CAD2Graph 的几何喂给他们的三角化/图构建函数**（复用全部算法代码，只替换几何来源）。

---

## 2. 已逐行验证的数据契约

### 2.1 pkl 结构（`merge_val_phase_V10.pkl`，实测样本 `1034`）

```python
pkl[sample_id] = [merge_segWalls, merge_fplanVeno]      # graphCrune 的合并输出版

merge_segWalls = {
  'vertices':        (2305, 2) float64   # 三角剖分全部顶点（**已 ×scale_coeff**）
  'segments':        list[1067] of (i,j) # 墙 PSLG 边
  'triangles_label': (432, 1) float64    # ★432 = **合并后区域数**，不是三角形数
  'triangles_area':  (432, 1) float64    # ★同上，按区域
  'edge':            list[1067] of (i,j)
  'edge_attr':       list[1067] of (wall_type, partition)
  'edge_dual':       list[1067] of (i,j)
  'inner2bd_index':  list[3334] of (i,j) # region → 顶点/三角形 的偏移索引边（febd 图用）
}
merge_fplanVeno = {
  'vertices':      (432, 2)   # 每个区域的**算术平均**质心
  'x':             (432, 2)   # 与 vertices 同一对象
  'segments':      list 994
  'segment_attr':  list 994 of [x1,y1,x2,y2]
  'segments_type': list 994 of int
  'segments_dual': list 994 of (i,j)
  'scale_coeff':   50
  'merge2tri':     dict[432]  # 区域 id -> [三角形 id, …]   ★输出端要用
}
```

**没有 `triangles` 键！** 合并后的 pkl **不保留三角形→顶点的表**，所以：

> ⚠️ **光靠 pkl 无法还原房间多边形** → 输出端必须**由我们自己保留三角剖分表** → **输入桥必须由我们自己生成 pkl**（不能靠"喂作者的数据然后读预测"来打通输出端）。

### 2.2 图像输入（实测 `img_dir/val/1034.png`：512×512 RGBA）

用 **matplotlib 描边绘制**（不是 cairosvg）：

| 元素 | 来源 | 颜色 | 线宽 |
|---|---|---|---|
| 背景 | mpl 默认 | 白 | — |
| 墙 PSLG | `walls['vertices'] / ['segments']` → `triPlot.plot` | **黑** | 0.5 |
| **窗** | `pDoors` 槽位（注意下面 §3.1 的互换） | **黑** | 0.5 |
| **门** | `pWindows` 槽位 | **蓝** | 0.5 |
| 画布边框矩形 | 4 个边界角点注入 `verticesDict` | 黑 | 0.5 |
| 三角剖分全部边 | `segWalls['segments']`（红层，**仓库里的脚本没有这一步**，是发布版多出来的） | **红** | — |

- 坐标空间：**annotation-pixel × `scale_coeff`(=50)**；`utils/triPlot.py` 内部 `ax.invert_yaxis()` 做 y 翻转。
- 画布尺寸来自"参考标注图"的尺寸；发布版还做了**等比例缩放 + 留白到 512×512**（`img_dir` 与 `ann_dir` 的留白带完全一致可证）。
- 下游：`ToTensor → Resize(256) → Normalize(ImageNet)`，所以精确像素尺寸不敏感，但**外观（颜色/线宽/留白比例）会影响 ResNet 特征**。

### 2.3 ★区域合并与标签无关（决定性的好消息）

`Utils/graphicsUtilsRe.py::graphCrune`：

- `regionGrow` 的分组判据**纯粹是拓扑**：只把「属于 `walls['segments']`（原始墙）的边」当作屏障，其余（Delaunay 内部边）可以穿过。**完全不读标签**。
- 标签只在**分组之后**用到两处：① 区域的面积加权多数标签；② 相邻区域标签是否相同 → `PARTITION/NOTPARTITION`。
- 因此**全 255（IGNORE）或全 0 标签时**，`merge2tri`、`merge_fplanVeno['vertices'|'x'|'segments'|'segment_attr'|'segments_dual']`、`merge_segWalls['segments'|'edge'|'edge_dual'|'inner2bd_index']` **与完美标签逐位相同**，只有 `triangles_label` / `triangles_area` / `edge_attr[:,1]` 不同。
- `edge_label`(partition) 在模型里是 **aux 目标**而非输入 → **推理可用合成标签，不影响主输出**。

---

## 3. 必须照抄的三个坑

### 3.1 门/窗在解析层被互换
`SVGParserCUBI.getWallShape`（`SvgProcessing_CubiCasa.py:179-182`）把 `<g id="Window">` 塞进 `primitiveDoors`、`<g id="Door">` 塞进 `primitiveWindows`。后果：`DOORTYPE(2)` 实际对应**窗**、`WINDOWTYPE(3)` 对应**门**；绘图时 `pDoors` 画黑、`pWindows` 画蓝。

→ **我们必须复刻这个互换**：把**窗**放进 `pDoors` 槽、**门**放进 `pWindows` 槽，否则 `segments_type` 的特征语义与训练时不一致。

### 3.2 合并用的是「原始墙」，但墙延长线也算屏障
`extendCornerWall` 就地 `segs.extend(newSegs)`，所以**延长线**虽然 `_genTriangleGraph` 里标为 `NOTWALL`，在 `graphCrune` 里却带 `ORIGINAL_WALL` → **会阻断区域合并**。必须与训练时完全一致地调用。

### 3.3 `graphCrune` 里一个潜在 bug（对我们有利）
`graphicsUtilsRe.py:414-423` 的判断 `if new_ET_Pairs[v[0]][-1] == PARTITION` 写在循环内、用 `v[0]` 而非 `sub_v`，是循环不变量 → `segments_dual` 恒等于 `v[0]`。结论：**标签派生出的 PARTITION 标志无法影响对偶映射**，进一步保证合成标签安全。

---

## 4. 架构

两个 Python 环境（必须隔离）：

- `cadruler`（3.13）：CAD2Graph 主流程，有 shapely；**不放 torch**。
- `vecfloorseg`（3.8）：VecFloorSeg + 他们的几何工具（`triangle`/numpy/matplotlib）。

```
CAD2Graph (cadruler)
  walls/doors/windows ──► 几何 JSON（mm）+ 画布参数
        │
        ▼  subprocess：vecfloorseg python
  src/spatial/vecfloorseg/build_dataset.py        ← 跑在 vecfloorseg 环境
      · mm → 像素（含 y 翻转）→ verticesDict（PSLG）
      · 注入 4 个边界角点
      · isLineIntersection → extendCornerWall → tr.triangulate        （复用他们的）
      · 合成标签（255=ignore）+ 鞋带公式面积                            （替代依赖 skimage 的投票）
      · _genTriangleGraph → buildDualRelationship → graphCrune        （复用他们的）
      · 写出：merge_<split>_phase_V10.pkl / <split>.txt / img_dir/<split>/<id>.png / triangles.npz
        │
        ▼  subprocess：vecfloorseg python
  graphgym/main.py --cfg CUBI.yaml --eval … （单样本临时数据集）
        │
        ▼  读 val_result.pkl
  src/spatial/vecfloorseg/postprocess.py
      · pred[region] + merge2tri + triangles.npz → 同标签三角形合并 → 多边形
      · 像素 → mm（逆变换）→ list[SpatialContour]
        │
        ▼  cadruler
  src/spatial/contours/vecfloorseg.py（实现 ISpatialContourExtractor）
```

### 关键约束（来自 `ISpatialContourExtractor` 契约）

1. `__init__` **必须**接受 CDT 的那 8 个默认参数（`min_area_mm2/erode_mm/min_width_mm/min_compactness/min_solidity/virtual_blocker_*`），否则 `create_contour_extractor()` 会因 `DEFAULT_CONTOUR_PARAMS` 恒被合并而 `TypeError`。
2. `get_visualization_data()` 必须返回**恰好 4 项**；`points` 为空则 `plot_floor_plan` 会**静默不产出 PNG**。
3. `geometry` 是**闭合外环的扁平 (x,y) 列表**（`.exterior.coords` 风格），单位 **mm**；**没有** `is_exterior` 字段，外部区域直接丢弃。
4. `visualization_suffix = "vecfloorseg"`，且 PNG 必须叫 `<drawing>_vecfloorseg.png`（web UI 按这个名字找）。
5. 必须在 `src/spatial/contours/__init__.py` 里 import 以触发注册。

---

## 5. 分阶段落地

| 阶段 | 内容 | 验收 | 状态 |
|---|---|---|---|
| **A** | `build_dataset.py`（几何→pkl+png，复用他们的函数） | 生成的 pkl 键/形状与作者样本逐一对应；`genGraph` 不报错 | ✅ 已完成 |
| **B** | `runner.py`（临时数据集 + `--eval` 子进程 + 读 `val_result.pkl`） | 用本地 1-epoch ckpt 跑出 400 维预测 | ✅ 已完成 |
| **C** | `postprocess.py`（pred+merge2tri+三角形表 → mm 多边形） | 在作者 CUBI 样本上叠加可视化，形状合理 | ✅ 已完成 |
| **D** | `contours/vecfloorseg.py` + 注册 + `settings.yaml` | `spatial.contour.algorithm: VecFloorSeg` 走通，产出 `<drawing>_vecfloorseg.png` | ✅ 已完成 |
| **E** | 精度调优（图像外观、画布留白比例、门窗槽位、`extendCornerWall` 参数） | 与 CDT/RGP 横向对比 | ⚠️ 已对比，**数字很差，根因是权重欠训练**（§6.5） |

> ⚠️ **前置依赖**：`build_dataset.py` 需要 `triangle`（`tr.triangulate`）。服务器上的 `setup_vecfloorseg_server.sh` 默认**不装**它（它只在 `--with-preproc-deps` 里）——适配层跑之前要补：`pip install -i <mirror> "triangle==20220202"`。

## 6. 验证记录（实测）

### 6.1 输入桥
合成户型 → pkl 的键名/形状与作者契约逐一对应；`triangles_label` 按区域而非按三角形；`merge2tri` 覆盖全部三角形。

### 6.2 模型链路
用**我们自己造**的 pkl+PNG 被作者的 `main.py` 接受（`dim_node_out: 12`、`num_splits: 3`、42.5M 参数），前向成功并产出 `val_result.pkl`。

### 6.3 真图纸端到端（`input_data/svg/2suite (1).svg`）
```
[VecFloorSeg] 1291 regions -> N contours
✅ Spatial reconstruction visualization saved to: output/viz/2suite (1)_vecfloorseg.png
🎨 Rendering graph: N spaces, 22 doors, 90 functional elements
```
整条流水线（元素提取 → 拓扑 → VecFloorSeg 轮廓 → 语义富化 → JSON-LD → 可视化）无报错。

**坐标往返正确**：输出轮廓的 `geo:asWKT` 包围盒落在图纸坐标系内，`props:hasArea` 与鞋带公式算出的面积逐位一致。

| 轮廓 | 顶点 | x 范围 (mm) | y 范围 (mm) | 面积 | 来源 |
|---|---|---|---|---|---|
| `Space_001` | 64 | 2805 – 30132 | 3139 – 30466 | 746.79 m² | 类别 0（背景/室外，覆盖整个画布） |
| `Space_002` | 44 | 8206 – 10292 | 16509 – 21749 | 5.20 m² | 类别 2（墙） |
| `Space_003` | 117 | 10345 – 21041 | 11749 – 22552 | 40.68 m² | 类别 2（墙） |

> 背景类 0 之所以覆盖整个画布（27.3 m × 27.3 m），是因为 `extendCornerWall` 会把墙端点**延伸到图像边界**，凸包因此等于画布。这与作者的设计一致，也正是 `DEFAULT_ROOM_LABELS` 只取 `{3,4,5,6,7,9,10,11}` 的原因。

> 上面的 N 取决于权重：本地 1-epoch ckpt 只预测 `{0, 2}`，所以须临时把 `keep_labels` 放宽到 `[0, 2]` 才能看到上面三个轮廓；默认配置下 N = 0。换成训练完的 ckpt 后 N 应为真实房间数。

### 6.4 配置接线
`scripts/check_vecfloorseg_config.py` 做 8 项静态自检（注册表 / settings.yaml / 工厂实例化 / 机器路径存在性 / 参数 / 可视化 4 元组 / 共享过滤器 / `algorithm_params` 合并），秒级返回。

**自检发现并修复的一个真 bug**：`algorithm_params.<NAME>` 原先只在 `settings.yaml` 的 `algorithm:` 选中该算法时才合并；从 CLI / Web UI 运行期参数覆盖算法名时，那一组专属参数会被**静默忽略**（`RGP` 一直中招）。已改为按**最终**算法名调 `SpatialPipelineConfig.contour_params_for()` / `classifier_params_for()` 取参，自检第 8 项专门守护这个行为。

### 6.5 基准矩阵实测结果（39 张图）

适配层已接入 `docs/benchmark_protocol.md` 的三层矩阵：

| 口径 | 结果 |
|---|---|
| 任务 1 · 轮廓（`contour_VecFloorSeg`） | 65 个空间 · 数量比 0.034 · 覆盖率 **0.013** · 一对一率 0.000 · mIoU 0.074 |
| 端到端最好的一格（`VecFloorSeg + LLMMultiStage`） | **0.030** |

对比 `CDT` 0.547 / `RGP` 0.656 的覆盖率，这一行基本是空的。

**根因是权重，不是适配层。** `best1.ckpt` 的 `best` 落在 **epoch 1**：1291 个 region
收敛成 1 条轮廓（标签分布 `0=1083, 3=2, 11=206`，房间类命中率 16.1%）。用本地
`0.ckpt` 做对照（`0=1126, 2=165`，房间类命中率 0.0%）得到 **0 条轮廓** ——
两者行为一致地"跟着权重走"，证明**管线是通的，是权重欠训练**。

→ 下一步是**按论文配方重训到收敛**（`max_epoch 200`，见
`docs/vecfloorseg_baseline_plan.md` §10.4），而不是继续调适配层。

## 7. 已知风险

1. **图像外观差异**：发布版的红层（三角形边）与 512 留白比例我们只能逼近，会让 ResNet 特征分布偏移 → 精度下降（但链路能跑通）。
2. **画布约定**：作者的画布 = CubiCasa 标注图尺寸（原图带留白）。我们按图纸 bbox + 边距自定画布，属于**分布偏移**；阶段 E 需要对标调整。
3. **`get_points`/`get_polygon` 轴序不一致**（`Utils/svgUtils.py:264` vs `:288`）——我们不走 SVG 解析，可绕过。
4. **门/窗互换**（§3.1）若复制错，`segments_type` 语义反了，精度会掉但不会崩。
5. `graphCrune` 依赖 `walls`（第二次 `isLineIntersection` 后的结果）与 `wholeEdgeSet`，参数顺序不能错。

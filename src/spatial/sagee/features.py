"""CAD2Graph 房间图 → 定长数值特征（8 维节点 / 5 维边）。

为什么是 8 / 5
--------------
SAGE-E 的架构是围绕"8 维节点 + 5 维边"设计的（见 ``docs/sagee_baseline_plan.md`` §3.1）。
这里**保持同样的宽度**，以便：

1. 端口与作者实现逐行可比（阶段 A 已在 RoomGraph 上验证复现）；
2. 不需要为了我们的特征去改模型结构（4 层、15,394 参数不变）。

但**特征定义是我们自己的**：我们不合并 RoomGraph（二者无可复现的公共特征 schema），
所以按 CAD2Graph 能算出来的量来设计。

设计原则
--------
* **文本无关**。这是 SAGE-E 相对 ``LLMMultiStage`` / ``TextMatching`` 的比较价值所在：
  它不看图纸上的任何文字，只用几何 + 拓扑。因此特征里**不含**任何文字或语义标签信息。
* **量纲统一**。长度用米、面积用平方米，连续量都做 clip，避免个别极端值主导训练。
* **可复现**。所有特征都能从 ``bot:`` 图谱（或等价地，从轮廓 + 构件几何）算出来。

节点特征（8）
-------------
===== ====================== ====================================================
序号  名称                   含义
===== ====================== ====================================================
0     ``log_area``           ``log1p(面积 m²)`` —— 压缩长尾
1     ``area_ratio``         面积 / 本图总面积 —— **相对大小是关键信号**（客厅通常最大）
2     ``perimeter_m``        周长（m）
3     ``short_side_m``       最小外接矩形短边（m）—— 反映"是不是窄条"
4     ``aspect_ratio``       长宽比（长边/短边，clip 到 10）
5     ``n_doors``            参与连接的门数量
6     ``n_windows``          关联的窗数量
7     ``n_furniture``        包含的家具构件数量（行为线索，如床/马桶/灶台）
===== ====================== ====================================================

边特征（5）
-----------
===== ====================== ====================================================
序号  名称                   含义
===== ====================== ====================================================
0     ``is_door``            1 = 有门连通（``bot:interfaceOf``）
1     ``is_window``          1 = 有窗连通
2     ``boundary_len_m``     共享边界长度（m，clip 到 20）
3     ``boundary_ratio``     共享边界长度 / 两房间周长的较小值
4     ``center_dist_m``      质心距离（m，clip 到 30）
===== ====================== ====================================================
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    'NODE_FEATURE_NAMES',
    'EDGE_FEATURE_NAMES',
    'NODE_FEATURE_DIM',
    'EDGE_FEATURE_DIM',
    'MAX_ASPECT_RATIO',
    'MAX_BOUNDARY_M',
    'MAX_CENTER_DIST_M',
    'short_side_mm',
    'polygon_metrics',
    'node_feature_vector',
    'edge_feature_vector',
]

NODE_FEATURE_NAMES: tuple[str, ...] = (
    'log_area', 'area_ratio', 'perimeter_m', 'short_side_m',
    'aspect_ratio', 'n_doors', 'n_windows', 'n_furniture',
)
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    'is_door', 'is_window', 'boundary_len_m', 'boundary_ratio', 'center_dist_m',
)
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

#: 长宽比上限（超过视为极端窄条，再大也没有额外信息量）
MAX_ASPECT_RATIO = 10.0
#: 共享边界长度上限（m）
MAX_BOUNDARY_M = 20.0
#: 质心距离上限（m）
MAX_CENTER_DIST_M = 30.0


def short_side_mm(poly: Any) -> float:
    """最小外接矩形短边（mm）。退化几何返回 0。"""
    try:
        rect = poly.minimum_rotated_rectangle
        coords = list(rect.exterior.coords)
    except Exception:                                            # noqa: BLE001
        return 0.0
    if len(coords) < 4:
        return 0.0
    sides = [math.dist(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    sides = [s for s in sides if s > 0]
    return float(min(sides)) if sides else 0.0


def polygon_metrics(poly: Any) -> dict[str, float]:
    """一次性算出该多边形参与所有特征计算的量。"""
    if poly is None or getattr(poly, 'is_empty', True):
        return {'area_m2': 0.0, 'perimeter_m': 0.0, 'short_side_m': 0.0,
                'aspect_ratio': 1.0, 'centroid': None}
    area_m2 = float(poly.area) / 1e6
    perimeter_m = float(poly.length) / 1000.0
    s_side = short_side_mm(poly) / 1000.0
    long_side = perimeter_m / 2.0 - s_side            # 矩形近似：P/2 = a + b
    if s_side <= 1e-9:
        aspect = 1.0
    else:
        aspect = max(1.0, min(MAX_ASPECT_RATIO, max(long_side, s_side) / s_side))
    return {
        'area_m2': area_m2,
        'perimeter_m': perimeter_m,
        'short_side_m': s_side,
        'aspect_ratio': aspect,
        'centroid': poly.centroid,
    }


def node_feature_vector(metrics: dict[str, float], total_area_m2: float,
                        n_doors: int, n_windows: int, n_furniture: int) -> np.ndarray:
    """按 :data:`NODE_FEATURE_NAMES` 的顺序拼出 8 维向量。"""
    area = float(metrics.get('area_m2', 0.0))
    ratio = (area / total_area_m2) if total_area_m2 > 1e-9 else 0.0
    return np.asarray([
        math.log1p(max(area, 0.0)),
        ratio,
        float(metrics.get('perimeter_m', 0.0)),
        float(metrics.get('short_side_m', 0.0)),
        float(metrics.get('aspect_ratio', 1.0)),
        float(n_doors),
        float(n_windows),
        float(n_furniture),
    ], dtype=np.float64)


def edge_feature_vector(is_door: bool, is_window: bool, boundary_len_mm: float,
                        min_perimeter_m: float, center_dist_mm: float) -> np.ndarray:
    """按 :data:`EDGE_FEATURE_NAMES` 的顺序拼出 5 维向量。"""
    boundary_m = min(float(boundary_len_mm) / 1000.0, MAX_BOUNDARY_M)
    ratio = (boundary_m / min_perimeter_m) if min_perimeter_m > 1e-9 else 0.0
    dist_m = min(float(center_dist_mm) / 1000.0, MAX_CENTER_DIST_M)
    return np.asarray([
        1.0 if is_door else 0.0,
        1.0 if is_window else 0.0,
        boundary_m,
        min(max(ratio, 0.0), 1.0),
        dist_m,
    ], dtype=np.float64)


def describe_schema() -> Sequence[dict[str, Any]]:
    """给文档 / 自检脚本用的 schema 描述。"""
    return [
        {'kind': 'node', 'index': i, 'name': n} for i, n in enumerate(NODE_FEATURE_NAMES)
    ] + [
        {'kind': 'edge', 'index': i, 'name': n} for i, n in enumerate(EDGE_FEATURE_NAMES)
    ]

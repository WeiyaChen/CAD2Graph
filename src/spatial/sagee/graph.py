"""把 CAD2Graph 的房间/构件记录组装成 SAGE-E 能吃的图。

本模块只做**组装**：几何 → 节点/边 → 定长特征（特征定义在 :mod:`.features`）。
两个入口共用同一套逻辑，保证「训练用的图」与「推理时的图」逐列同构：

* :func:`from_jsonld` —— 从 ``bot:`` 图谱字典解析（离线生成训练集用）；
* :func:`from_domain` —— 从 ``SpatialContour`` / ``SpatialComponent`` 领域对象解析
  （分类器运行时用）。

⚠️ 节点顺序即输出顺序，必须**稳定**：调用方（分类器）按同一顺序回写预测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads

from .features import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    edge_feature_vector,
    node_feature_vector,
    polygon_metrics,
)

__all__ = ['RoomRecord', 'ElementRecord', 'RoomGraph', 'from_jsonld', 'from_domain',
           'build_room_graph']

#: 判定"共享边界"的容差（mm）。同一套轮廓提取产出的相邻房间通常精确共边，
#: 留一点容差是为了兼容浮点误差与轮廓被轻微简化的情况。
BOUNDARY_TOL_MM = 100.0

#: 认为是"门窗"的构件类别名（``beo:`` 之后的部分）
DOOR_CATEGORIES = frozenset({'Door'})
WINDOW_CATEGORIES = frozenset({'Window'})


# --------------------------------------------------------------------------- #
# 规范化中间表示
# --------------------------------------------------------------------------- #


@dataclass
class RoomRecord:
    """一个空间轮廓的规范化记录。"""

    id: str
    poly: Polygon | None = None
    area_sqm: float | None = None
    neighbor_ids: list[str] = field(default_factory=list)
    element_ids: list[str] = field(default_factory=list)
    label: str | None = None


@dataclass
class ElementRecord:
    """一个构件的规范化记录。"""

    uid: str
    category: str = ''
    interface_of: list[str] = field(default_factory=list)
    poly: Polygon | None = None


@dataclass
class RoomGraph:
    """一张图纸的房间图。"""

    node_ids: list[str]
    node_feat: np.ndarray                 # (N, NODE_FEATURE_DIM)
    edge_index: np.ndarray                # (2, E) —— 第 0 行 src，第 1 行 dst
    edge_feat: np.ndarray                 # (E, EDGE_FEATURE_DIM)
    labels: list[str | None] = field(default_factory=list)
    source: str = ''
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1]) if self.edge_index.size else 0

    def __repr__(self) -> str:                                   # pragma: no cover
        return ('RoomGraph(%s: %d 房间, %d 边, dim=%d/%d)'
                % (self.source or '?', self.n_nodes, self.n_edges,
                   self.node_feat.shape[1] if self.node_feat.size else 0,
                   self.edge_feat.shape[1] if self.edge_feat.size else 0))


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #


def _as_polygon(value: Any) -> Polygon | None:
    """接受 WKT 字符串 / ``{"@value": wkt}`` / shapely 多边形 / 坐标环。"""
    if value is None:
        return None
    if isinstance(value, Polygon):
        return value
    if isinstance(value, Mapping):
        value = value.get('@value')
        if value is None:
            return None
    if isinstance(value, str):
        try:
            geom = wkt_loads(value)
        except Exception:                                        # noqa: BLE001
            return None
        return geom if isinstance(geom, Polygon) and not geom.is_empty else None
    try:
        return Polygon(value)
    except Exception:                                            # noqa: BLE001
        return None


def build_room_graph(rooms: Sequence[RoomRecord],
                     elements: Sequence[ElementRecord] = (),
                     *,
                     source: str = '',
                     directed: bool = True) -> RoomGraph:
    """把规范化记录组装成图。

    Args:
        rooms: 房间记录，**顺序即节点顺序**。
        elements: 构件记录（门窗用于判边类型，其余用于数家具）。
        source: 来源标识（写进 ``stats``，便于溯源）。
        directed: ``True`` 时每条邻接同时生成正反两条有向边 —— 与作者实现一致
            （他们的图是双向的，而 SAGE-E 只沿入边聚合）。
    """
    rooms = list(rooms)
    elements = list(elements)
    elem_by_uid = {e.uid: e for e in elements}

    index_of = {r.id: i for i, r in enumerate(rooms)}
    metrics = [polygon_metrics(r.poly) for r in rooms]
    # 面积优先用图谱里的 props:hasArea（几何缺失时也能用），否则回退到几何计算
    areas = [
        float(r.area_sqm) if r.area_sqm not in (None, 0) else metrics[i]['area_m2']
        for i, r in enumerate(rooms)
    ]
    total_area = float(sum(areas))

    # ---- 节点特征 ---------------------------------------------------------- #
    node_feat = np.zeros((len(rooms), NODE_FEATURE_DIM), dtype=np.float64)
    n_doors_total = n_windows_total = n_furniture_total = 0
    for i, room in enumerate(rooms):
        n_door = n_win = n_furn = 0
        for uid in room.element_ids:
            cat = elem_by_uid[uid].category if uid in elem_by_uid else ''
            if cat in DOOR_CATEGORIES:
                n_door += 1
            elif cat in WINDOW_CATEGORIES:
                n_win += 1
            else:
                n_furn += 1
        n_doors_total += n_door
        n_windows_total += n_win
        n_furniture_total += n_furn
        m = dict(metrics[i])
        m['area_m2'] = areas[i]
        node_feat[i] = node_feature_vector(m, total_area, n_door, n_win, n_furn)

    # ---- 边 ---------------------------------------------------------------- #
    # 1) 房间对 -> 连接它们的门窗
    pair_openings: dict[tuple[int, int], dict[str, int]] = {}
    for elem in elements:
        cat = elem.category
        is_door = cat in DOOR_CATEGORIES
        is_win = cat in WINDOW_CATEGORIES
        if not (is_door or is_win):
            continue
        if is_door:
            # 门：靠 bot:interfaceOf 明确指明它连的两个房间
            targets = [index_of[t] for t in elem.interface_of if t in index_of]
        else:
            # 窗：靠"被哪些房间包含"（builder 用膨胀 300mm 的房间做判定，
            # 所以一扇窗可能同时属于相邻的两个房间）
            targets = [i for i, r in enumerate(rooms) if elem.uid in set(r.element_ids)]
        for a in range(len(targets)):
            for b in range(a + 1, len(targets)):
                key = (min(targets[a], targets[b]), max(targets[a], targets[b]))
                slot = pair_openings.setdefault(key, {'door': 0, 'window': 0})
                slot['door' if is_door else 'window'] += 1

    # 2) 邻接对 -> 边（用几何算共享边界；没有几何就退化为 0）
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for i, room in enumerate(rooms):
        for nb in room.neighbor_ids:
            j = index_of.get(nb)
            if j is None or j == i:
                continue
            key = (min(i, j), max(i, j))
            info = edges.setdefault(key, {'boundary_mm': 0.0, 'dist_mm': 0.0})
            if info['dist_mm'] or not info['boundary_mm']:
                a_poly, b_poly = rooms[key[0]].poly, rooms[key[1]].poly
                if a_poly is not None and b_poly is not None:
                    try:
                        shared = a_poly.boundary.intersection(
                            b_poly.boundary.buffer(BOUNDARY_TOL_MM))
                        info['boundary_mm'] = max(info['boundary_mm'], float(shared.length))
                        info['dist_mm'] = float(a_poly.centroid.distance(b_poly.centroid))
                    except Exception:                            # noqa: BLE001
                        pass

    keys = sorted(edges)
    rows, cols, feats = [], [], []
    for (i, j) in keys:
        info = edges[(i, j)]
        openings = pair_openings.get((i, j), {})
        min_perim = min(metrics[i]['perimeter_m'], metrics[j]['perimeter_m'])
        vec = edge_feature_vector(
            is_door=openings.get('door', 0) > 0,
            is_window=openings.get('window', 0) > 0,
            boundary_len_mm=info['boundary_mm'],
            min_perimeter_m=min_perim,
            center_dist_mm=info['dist_mm'],
        )
        rows.append(i)
        cols.append(j)
        feats.append(vec)
        if directed and i != j:
            rows.append(j)
            cols.append(i)
            feats.append(vec)

    edge_index = (np.asarray([rows, cols], dtype=np.int64) if rows
                  else np.zeros((2, 0), dtype=np.int64))
    edge_feat = (np.asarray(feats, dtype=np.float64) if feats
                 else np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float64))

    isolated = int(sum(1 for i in range(len(rooms)) if not any(i in k for k in keys)))
    graph = RoomGraph(
        node_ids=[r.id for r in rooms],
        node_feat=node_feat,
        edge_index=edge_index,
        edge_feat=edge_feat,
        labels=[r.label for r in rooms],
        source=source,
        stats={
            'n_nodes': len(rooms),
            'n_edges_undirected': len(keys),
            'n_edges': int(edge_index.shape[1]),
            'n_isolated_nodes': isolated,
            'total_area_m2': round(total_area, 2),
            'n_doors': n_doors_total,
            'n_windows': n_windows_total,
            'n_furniture': n_furniture_total,
            'n_rooms_without_geometry': sum(1 for m in metrics if m['centroid'] is None),
        },
    )
    return graph


# --------------------------------------------------------------------------- #
# 入口 1：bot: JSON-LD 图谱
# --------------------------------------------------------------------------- #


def _node_types(node: Mapping[str, Any]) -> list[str]:
    types = node.get('@type') or []
    return [types] if isinstance(types, str) else [str(t) for t in types]


def _short_id(ref: Any) -> str:
    """``{"@id": "inst:Space_001"}`` → ``"inst:Space_001"``。"""
    if isinstance(ref, Mapping):
        return str(ref.get('@id', ''))
    return str(ref)


def from_jsonld(graph_dict: Mapping[str, Any], *,
                space_type: str = 'bot:Space',
                element_type: str = 'bot:Element',
                label_prefix: str = 'bldg:',
                source: str = '',
                label_map: Mapping[str, str] | None = None) -> RoomGraph:
    """从 ``bot:`` 图谱字典构建房间图。

    ``label_map`` 存在时，用 ``@type`` 里的 ``bldg:X`` 作为节点标签并做映射
    （GT 图谱用它来产监督信号）；否则 ``label`` 置空（系统输出在富化前没有类型）。
    """
    nodes = graph_dict.get('@graph') or []
    rooms: list[RoomRecord] = []
    elements: list[ElementRecord] = []

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        types = _node_types(node)

        if space_type in types:
            label = None
            if label_map is not None:
                for t in types:
                    if t.startswith(label_prefix):
                        label = label_map.get(t[len(label_prefix):], t[len(label_prefix):])
                        break
            rooms.append(RoomRecord(
                id=str(node.get('@id', '')),
                poly=_as_polygon(node.get('geo:asWKT')),
                area_sqm=node.get('props:hasArea'),
                neighbor_ids=[_short_id(r) for r in (node.get('bot:adjacentZone') or [])],
                element_ids=[_short_id(r) for r in (node.get('bot:containsElement') or [])],
                label=label,
            ))
        elif element_type in types:
            category = next((t.split(':', 1)[1] for t in types if t.startswith('beo:')), '')
            elements.append(ElementRecord(
                uid=str(node.get('@id', '')),
                category=category,
                interface_of=[_short_id(r) for r in (node.get('bot:interfaceOf') or [])],
                poly=_as_polygon(node.get('geo:asWKT')),
            ))

    return build_room_graph(rooms, elements, source=source)


# --------------------------------------------------------------------------- #
# 入口 2：领域对象（分类器运行时）
# --------------------------------------------------------------------------- #


def from_domain(contours: Sequence[Any],
                components: Sequence[Any] = (),
                adjacency: Mapping[str, Sequence[str]] | None = None,
                *,
                source: str = '') -> RoomGraph:
    """从 ``SpatialContour`` / ``SpatialComponent`` 构建房间图。

    ⚠️ 构件必须带上 ``properties['node']``（由 ``SemanticEnricher._build_components``
    注入），否则拿不到 ``bot:interfaceOf``，边类型会退化成"全部为墙连接"。
    """
    rooms: list[RoomRecord] = []
    for contour in contours:
        poly = _as_polygon(contour.geometry)
        node = (contour.attributes or {}).get('node') or {}
        neighbor_ids = list((adjacency or {}).get(contour.id) or contour.neighbors or [])
        rooms.append(RoomRecord(
            id=contour.id,
            poly=poly,
            area_sqm=contour.area_sqm,
            neighbor_ids=[str(n) for n in neighbor_ids],
            element_ids=[str(e) for e in (contour.elements or [])],
        ))

    elements: list[ElementRecord] = []
    for comp in components:
        node = (comp.properties or {}).get('node') or {}
        elements.append(ElementRecord(
            uid=str(comp.uid),
            category=str(comp.category or ''),
            interface_of=[_short_id(r) for r in (node.get('bot:interfaceOf') or [])],
            poly=_as_polygon(comp.geometry) if comp.geometry is not None
            else _as_polygon(node.get('geo:asWKT')),
        ))

    return build_room_graph(rooms, elements, source=source)

"""把 **GT（人工标注）的空间轮廓** 直接当作提取结果 —— 用于「固定轮廓变量」的消融。

为什么需要
----------
比较空间类型识别算法时，如果每个被测分类器跑在**各自**的轮廓上，差异里就混进了
轮廓提取的误差，不能算有效的 ablation。本提取器把起点换成 GT 空间轮廓，使
「空间类型识别」这一项只剩分类器本身一个变量。

两条硬规矩
----------
1. **只取几何，绝不读 GT 的类型。** GT 的 ``@type`` 里带着正确答案
   （``bldg:Bedroom`` 等）；一旦泄漏给被测分类器，测的就不是分类能力。
   所有轮廓的 ``label`` 一律写 ``"Unknown"``。
2. **文字与构件仍来自系统。** GT 只标注空间，没有单个构件的标注（标注量太大），
   所以 ``texts`` / ``components`` 原样来自 ``SymPointV2 → svg_parser`` 这一链，
   本提取器完全不用它们，只消费墙体几何做坐标对齐诊断。

怎么找到 GT
-----------
``extract()`` 需要知道"这是哪张图纸"。该信息由 :mod:`src.topology.builder` 通过
``context`` 传入（以前这个参数根本没被传下去，是本模块引入的管线上的一处补充）：

    context = {'base_name': '2suite (1)', 'json_output_path': ...}

命名映射：``2suite (1)`` ↔ ``2suite_annotated (1)_gt.jsonld``。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.wkt import loads as wkt_loads

from ..contracts import ISpatialContourExtractor
from ..domain import SpatialComponent, SpatialContour
from ..registry import CONTOUR_EXTRACTORS

__all__ = ['GTContourExtractor', 'find_gt_jsonld', 'load_gt_polygons']

#: GT 轮廓的编号前缀（与 RGP / CDT 的 ``Space_%03d`` 保持一致）
_ID_FMT = 'Space_%03d'


def find_gt_jsonld(base_name: str, gt_dir: str | None = None) -> str | None:
    """按系统图纸名找到对应的 GT 标注文件。

    镜像 :meth:`src.web_ui_server.UIHandler._find_gt_for_base` 的查找顺序，保持
    网页端与实验脚本指向同一份 GT：

    1. ``<base>_gt.jsonld`` / ``<base>_annotated_gt.jsonld`` / ``<base>_Annotated_gt.jsonld``
    2. 遍历 ``*_gt.jsonld``，把文件名里的 ``_gt`` / ``_annotated`` 去掉后与 ``base`` 比对
    """
    import os

    if not base_name:
        return None
    if gt_dir is None:
        from src.config.config import settings
        gt_dir = str(settings.gt_dir)

    for name in ('%s_gt.jsonld' % base_name,
                 '%s_annotated_gt.jsonld' % base_name,
                 '%s_Annotated_gt.jsonld' % base_name):
        p = os.path.join(gt_dir, name)
        if os.path.isfile(p):
            return p

    if not os.path.isdir(gt_dir):
        return None
    for fn in sorted(os.listdir(gt_dir)):
        if not fn.endswith('_gt.jsonld'):
            continue
        stem = fn[:-len('_gt.jsonld')]
        stem = stem.replace('_annotated', '').replace('_Annotated', '')
        if stem == base_name:
            return os.path.join(gt_dir, fn)
    return None


def load_gt_polygons(gt_path: str) -> list[Polygon]:
    """从 GT 图谱里取出**空间多边形**（只取几何，忽略类型）。"""
    import json

    with open(gt_path, encoding='utf-8') as f:
        data = json.load(f)
    polys: list[Polygon] = []
    for node in data.get('@graph') or []:
        if not isinstance(node, dict):
            continue
        types = node.get('@type') or []
        types = [types] if isinstance(types, str) else list(types)
        if 'bot:Space' not in types:
            continue
        wkt = node.get('geo:asWKT')
        if isinstance(wkt, dict):
            wkt = wkt.get('@value')
        if not wkt:
            continue
        try:
            geom = wkt_loads(wkt)
        except Exception:                                        # noqa: BLE001
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif hasattr(geom, 'geoms'):                             # MultiPolygon 等
            polys.extend(g for g in geom.geoms
                         if isinstance(g, Polygon) and not g.is_empty)
    return polys


def _bbox(polys) -> tuple[float, float, float, float] | None:
    if not polys:
        return None
    try:
        b = unary_union(polys).bounds
    except Exception:                                            # noqa: BLE001
        return None
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _bbox_iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
    return inter / (area_a + area_b - inter)


@CONTOUR_EXTRACTORS.register('GT', aliases=('GROUNDTRUTH', 'GT_CONTOUR'))
class GTContourExtractor(ISpatialContourExtractor):
    """直接用 GT 空间轮廓，绕开轮廓提取算法（消融用）。"""

    name = 'GT'
    visualization_suffix = 'gt'

    def __init__(
        self,
        *,
        # —— 与 CDT / RGP 同名的共享参数：工厂会把它们合并进来（见 docs）——
        # 本算法不使用，但必须接受，否则运行时会 TypeError。
        min_area_mm2: float = 2000000.0,
        erode_mm: float = 250.0,
        min_width_mm: float = 600.0,
        min_compactness: float = 0.08,
        min_solidity: float = 0.25,
        virtual_blocker_dist_tol_mm: float = 300.0,
        virtual_blocker_max_len_mm: float = 3500.0,
        virtual_blocker_angle_tol_deg: float = 0.0,
        # —— 本算法专有 ——
        gt_dir: str | None = None,
        base_name: str | None = None,
        strict: bool = True,
        min_align_iou: float = 0.2,
    ):
        self.gt_dir = gt_dir or None
        #: 兜底图纸名；正常情况下由 ``context['base_name']`` 传入
        self.base_name = base_name or None
        #: True：找不到 GT 就报错。实验里要的是"响亮地失败"，而不是静默产出 0 个轮廓。
        self.strict = bool(strict)
        #: GT 与系统墙体包围盒的 IoU 下限，低于它认为坐标系没对齐
        self.min_align_iou = float(min_align_iou)
        # 共享参数仅为兼容而接受，这里显式丢弃（避免误用时静默生效）
        self._ignored = dict(
            min_area_mm2=min_area_mm2, erode_mm=erode_mm, min_width_mm=min_width_mm,
            min_compactness=min_compactness, min_solidity=min_solidity,
            virtual_blocker_dist_tol_mm=virtual_blocker_dist_tol_mm,
            virtual_blocker_max_len_mm=virtual_blocker_max_len_mm,
            virtual_blocker_angle_tol_deg=virtual_blocker_angle_tol_deg,
        )

        #: 供可视化读取（``plot_floor_plan`` 要求恰好 4 项且 points 非空）
        self.points: list[Any] = []
        self.real_walls: list[Any] = []
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ 公共入口
    def extract(
        self,
        *,
        walls: Sequence[Any] = (),
        doors: Sequence[Any] = (),
        windows: Sequence[Any] = (),
        texts: Sequence[Any] = (),
        components: Sequence[SpatialComponent] = (),
        context: Mapping[str, Any] | None = None,
    ) -> list[SpatialContour]:
        """读取 GT 空间多边形并原样返回。

        ``doors`` / ``windows`` / ``texts`` / ``components`` 一律不使用 —— 它们本来
        就来自系统，且这里只需要几何。
        """
        self.points = []
        self.real_walls = list(walls)
        self.stats = {}

        base_name = None
        if context:
            base_name = context.get('base_name')
        base_name = base_name or self.base_name
        if not base_name:
            raise RuntimeError(
                'GT 轮廓提取器不知道当前是哪张图纸：`context["base_name"]` 未传。\n'
                '它由 src/topology/builder.py 注入；若你是直接调用本类，请显式传 '
                'context={"base_name": "2suite (1)"} 或在构造时指定 base_name=…。')

        gt_path = find_gt_jsonld(base_name, self.gt_dir)
        if not gt_path:
            msg = ('找不到 %s 对应的 GT 标注（gt_dir=%s）。实验里不能静默跳过 —— '
                   '拿不到 GT 轮廓时分类器看到的就是别人的轮廓，结论会失真。'
                   % (base_name, self.gt_dir or '<settings.gt_dir>'))
            if self.strict:
                raise FileNotFoundError(msg)
            print('[GT] %s' % msg)
            return []

        polys = load_gt_polygons(gt_path)
        if not polys:
            msg = '%s 里没有可用的 bot:Space 多边形' % gt_path
            if self.strict:
                raise ValueError(msg)
            print('[GT] %s' % msg)
            return []

        # —— 坐标对齐诊断：GT 来自 *_annotated.dxf，系统几何来自 SVG 解析。
        # 两者理论上同坐标系，但一旦不同，"文字落在哪个空间里"会整体错位，
        # 而且是**静默**错位。这里显式量一下，低于阈值就大声警告。
        gt_box = _bbox(polys)
        wall_polys = []
        for geom in walls:
            try:
                wall_polys.extend(geom.geoms if hasattr(geom, 'geoms') else [geom])
            except Exception:                                    # noqa: BLE001
                continue
        align = _bbox_iou(gt_box, _bbox(wall_polys))
        self.stats = {'gt_path': gt_path, 'n_polygons': len(polys),
                      'gt_bbox': gt_box, 'align_iou': align, 'base_name': base_name}
        print('[GT] %s -> %d 个 GT 空间；bbox 与墙体对齐 IoU %.3f'
              % (base_name, len(polys), align))
        if wall_polys and align < self.min_align_iou:
            print('[GT] ⚠️ 坐标系可能没对齐（IoU %.3f < %.2f）：GT 来自 '
                  '*_annotated.dxf，系统几何来自 SVG。文字/构件落位会整体错位，'
                  '后续对比不可信 —— 先查清坐标系映射。'
                  % (align, self.min_align_iou))

        # 面积从大到小排序，保证多次运行结果稳定（与 RGP 一致）
        polys.sort(key=lambda p: (-round(p.area, 3), p.bounds[0], p.bounds[1]))
        contours = [
            SpatialContour(
                id=_ID_FMT % i,
                label='Unknown',          # ⚠️ 绝不写 GT 类型
                geometry=[tuple(c) for c in poly.exterior.coords],
            )
            for i, poly in enumerate(polys, start=1)
        ]
        self.points = [pt for c in contours for pt in c.geometry]
        return contours

    # ------------------------------------------------------------------ 可视化
    def get_visualization_data(self) -> Any:
        """返回 ``(real_walls, virtual_walls, triangles, points)``。

        ``plot_floor_plan`` 要求**恰好 4 项**且 ``points`` 非空（否则直接不落盘）。
        GT 没有三角网，所以第三项恒为 ``None``，左图退化为"墙体 + GT 节点"，
        正好方便肉眼核对 GT 是否落在正确的位置。
        """
        return (self.real_walls, [], None, self.points)

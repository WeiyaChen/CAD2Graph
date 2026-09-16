"""VecFloorSeg — 可插拔空间轮廓提取（第三个 baseline，与 CDT / RGP 并列）。

实现 ``ISpatialContourExtractor``。真正的计算发生在 **vecfloorseg** conda 环境
（PyTorch 1.13 + cu117 + 作者内联的 torch_geometric），本模块只在 CAD2Graph 的
``cadruler`` 环境里做编排与后处理，**绝不 import torch**。

为什么需要一个"桥"
------------------
VecFloorSeg 是"矢量化线框图 + 栅格图"的图神经网络：输入不是 CAD 图元，而是
1) 一张线框渲染图；2) 一个由受限 Delaunay 三角化导出、再按墙做区域合并后的图。
作者原始链路是 ``model.svg -> SVGParserCUBI -> ... -> pkl``，但那条链路只认
CubiCasa 的 SVG 约定。由于三角化真正需要的只是一个 PSLG，而 CAD2Graph 的
``walls`` 本身就是 PSLG，我们**跳过 SVG 解析**，直接把几何喂给作者的
``isLineIntersection / extendCornerWall / tr.triangulate / _genTriangleGraph /
buildDualRelationship / graphCrune``，从而在保证保真度的前提下支持任意 CAD 图纸。

详见 ``docs/vecfloorseg_adapter_design.md``（含逐行验证的数据契约与坑）。

配置
----
三个与机器绑定的项优先从环境变量读取（建议写在 ``.env``，不要提交绝对路径），
也都可以在 ``settings.yaml`` 的 ``spatial.contour.algorithm_params.VecFloorSeg``
里显式指定（显式值优先）：

============ ========================= ========================================
参数        环境变量                  缺省行为
============ ========================= ========================================
python_exe    ``VECFLOORSEG_PYTHON``     无，必须配置
checkpoint    ``VECFLOORSEG_CKPT``       无，必须配置
vf_root       ``VECFLOORSEG_ROOT``       ``<仓库>/third_party/VecFloorSeg``
============ ========================= ========================================

其余参数（``canvas_px`` / ``margin_ratio`` / ``wall_thickness_mm`` /
``simplify_tol_mm`` / ``keep_labels`` / ``device`` / ``timeout_s`` …）都有合理默认值。

⚠️ 两个易踩的点
* ``__init__`` **必须**接受 CDT 的那 8 个默认参数：``DEFAULT_CONTOUR_PARAMS``
  在 ``create_contour_extractor()`` 里恒被合并进来，少一个就会 ``TypeError``。
* ``get_visualization_data()`` 必须返回**恰好 4 项**，且 ``points`` 非空，否则
  ``plot_floor_plan`` 会静默不产出 PNG。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from ..contracts import ISpatialContourExtractor
from ..domain import SpatialContour
from ..registry import CONTOUR_EXTRACTORS

__all__ = ['VecFloorSegContourExtractor']

# 与 CDT 一致的构造参数名（见 contracts/registry 的合并机制）
_SHARED_PARAMS: dict[str, float] = {
    'min_area_mm2': 2000000.0,
    'erode_mm': 250.0,
    'min_width_mm': 600.0,
    'min_compactness': 0.08,
    'min_solidity': 0.25,
    'virtual_blocker_dist_tol_mm': 300.0,
    'virtual_blocker_max_len_mm': 3500.0,
    'virtual_blocker_angle_tol_deg': 0.0,
}


@CONTOUR_EXTRACTORS.register('VecFloorSeg', aliases=('VFS', 'VECTORFLOORSEG'))
class VecFloorSegContourExtractor(ISpatialContourExtractor):
    """用 VecFloorSeg（两流图注意力网络）预测房间轮廓。"""

    name = 'VecFloorSeg'
    visualization_suffix = 'vecfloorseg'

    def __init__(
        self,
        *,
        # ---- 与 CDT/RGP 共享的过滤器参数（必须存在，否则创建时 TypeError）----
        min_area_mm2: float = _SHARED_PARAMS['min_area_mm2'],
        erode_mm: float = _SHARED_PARAMS['erode_mm'],
        min_width_mm: float = _SHARED_PARAMS['min_width_mm'],
        min_compactness: float = _SHARED_PARAMS['min_compactness'],
        min_solidity: float = _SHARED_PARAMS['min_solidity'],
        virtual_blocker_dist_tol_mm: float = _SHARED_PARAMS['virtual_blocker_dist_tol_mm'],
        virtual_blocker_max_len_mm: float = _SHARED_PARAMS['virtual_blocker_max_len_mm'],
        virtual_blocker_angle_tol_deg: float = _SHARED_PARAMS['virtual_blocker_angle_tol_deg'],
        # ---- VecFloorSeg 专有参数 ----
        python_exe: str | None = None,
        checkpoint: str | None = None,
        vf_root: str | None = None,
        canvas_px: int = 512,
        margin_ratio: float = 0.08,
        wall_thickness_mm: float = 0.0,
        simplify_tol_mm: float = 0.0,
        keep_labels: Sequence[int] | None = None,
        work_root: str | None = None,
        keep_workdir: bool = False,
        timeout_s: int = 1800,
        device: str = 'cuda:0',
    ):
        self.shape_filter_params = dict(_SHARED_PARAMS)
        self.shape_filter_params.update(
            min_area_mm2=min_area_mm2, erode_mm=erode_mm,
            min_width_mm=min_width_mm, min_compactness=min_compactness,
            min_solidity=min_solidity,
        )
        self.min_area_mm2 = min_area_mm2
        self.python_exe = python_exe or os.environ.get('VECFLOORSEG_PYTHON', '')
        self.checkpoint = checkpoint or os.environ.get('VECFLOORSEG_CKPT', '')
        self.vf_root = vf_root or os.environ.get('VECFLOORSEG_ROOT') or ''
        self.canvas_px = int(canvas_px)
        self.margin_ratio = float(margin_ratio)
        self.wall_thickness_mm = float(wall_thickness_mm)
        self.simplify_tol_mm = float(simplify_tol_mm)
        self.keep_labels = list(keep_labels) if keep_labels else None
        self.work_root = work_root
        self.keep_workdir = bool(keep_workdir)
        self.timeout_s = int(timeout_s)
        self.device = device

        self._viz: tuple[Any, Any, Any, Any] = ([], [], None, [])
        self._last_error: str = ''

    # ------------------------------------------------------------------ API
    def extract(
        self,
        *,
        walls: Sequence[Any] = (),
        doors: Sequence[Any] = (),
        windows: Sequence[Any] = (),
        texts: Sequence[Any] = (),
        components: Sequence[Any] = (),
        context: Mapping[str, Any] | None = None,
    ) -> list[SpatialContour]:
        self._viz = (list(walls), [], None, [])
        self._last_error = ''
        if not walls:
            return []

        try:
            return self._extract_impl(list(walls), list(doors), list(windows))
        except Exception as exc:                       # 不让整条流水线挂掉
            self._last_error = '%s: %s' % (type(exc).__name__, exc)
            print('[VecFloorSeg] extraction failed -> %s' % self._last_error)
            self._viz = (list(walls), [], None, [])
            return []

    def get_visualization_data(self) -> Any:
        return self._viz

    # -------------------------------------------------------------- internals
    def _extract_impl(self, walls, doors, windows) -> list[SpatialContour]:
        from .filters import SpaceShapeFilter          # 与本模块同包（contours/）
        from ..vecfloorseg import geometry as vfs_geometry
        from ..vecfloorseg import postprocess as vfs_post
        from ..vecfloorseg import runner as vfs_runner

        if not self.python_exe or not os.path.isfile(self.python_exe):
            raise RuntimeError(
                'VecFloorSeg interpreter not configured/found (%r).  Set '
                'spatial.contour.algorithm_params.VecFloorSeg.python_exe or the '
                'VECFLOORSEG_PYTHON environment variable.' % self.python_exe)
        if not self.checkpoint or not os.path.isfile(self.checkpoint):
            raise RuntimeError(
                'VecFloorSeg checkpoint not configured/found (%r).  Set '
                'spatial.contour.algorithm_params.VecFloorSeg.checkpoint or the '
                'VECFLOORSEG_CKPT environment variable.' % self.checkpoint)

        work = tempfile.mkdtemp(prefix='vfs_', dir=self.work_root)
        dataset_dir = os.path.join(work, 'dataset')
        out_dir = os.path.join(work, 'results')
        sample_id = 'cad2graph'

        try:
            spec = vfs_geometry.build_spec(
                sample_id, walls, doors, windows,
                dataset_dir=dataset_dir, work_dir=work, split='val',
                canvas_px=self.canvas_px, margin_ratio=self.margin_ratio,
                wall_thickness_mm=self.wall_thickness_mm,
            )
            result = vfs_runner.run_adapter(
                spec, python_exe=self.python_exe, checkpoint=self.checkpoint,
                out_dir=out_dir, vf_root=self.vf_root or None,
                timeout_s=self.timeout_s, device=self.device,
            )

            result_pkl = vfs_post.find_result_pkl(result.run_dir, prefer='val')
            pred, _fname = vfs_post.load_predictions(result_pkl)
            table = vfs_post.TriangleTable(result.triangles_npz)
            contours, tri_px = vfs_post.regions_to_contours(
                table, pred, result.transform,
                keep_labels=self.keep_labels,
                simplify_tol_mm=self.simplify_tol_mm,
                # from_params 会忽略它不认识的共享参数键（virtual_blocker_*）
                shape_filter=SpaceShapeFilter.from_params(self.shape_filter_params),
            )

            # 可视化：第三项恒为 None（无网格可画，与 RGP 一致），
            # 第四项用三角形质心（mm），非空才会出图。
            centroids_px = tri_px.mean(axis=1)
            to_mm = vfs_post._make_to_mm(result.transform)
            points_mm = [to_mm(float(x), float(y)) for x, y in centroids_px]
            self._viz = (list(walls), [], None, points_mm)
            counts: dict[int, int] = {}
            for lb in pred:
                counts[int(lb)] = counts.get(int(lb), 0) + 1
            room = vfs_post.DEFAULT_ROOM_LABELS
            n_room = sum(c for lb, c in counts.items() if lb in room)
            print('[VecFloorSeg] %d regions -> %d contours' % (len(pred), len(contours)))
            print('[VecFloorSeg] 预测分布: %s'
                  % ', '.join('%d=%d' % (k, counts[k]) for k in sorted(counts)))
            print('[VecFloorSeg] 命中房间类 %s 的 region: %d/%d (%.1f%%)'
                  % (sorted(room), n_room, len(pred),
                     100.0 * n_room / max(1, len(pred))))
            return contours
        finally:
            if not self.keep_workdir:
                shutil.rmtree(work, ignore_errors=True)
            else:
                print('[VecFloorSeg] work dir kept at %s' % work)

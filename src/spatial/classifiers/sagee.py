# src/spatial/classifiers/sagee.py
"""SAGE-E 空间类型识别（第四个比较项，纯几何 + 拓扑，**不读图纸文字**）。

与 ``LLMMultiStage`` / ``TextMatching`` 的区别

======================  ============  ============  ==================
分类器                  用文字        用大模型       用几何/拓扑
======================  ============  ============  ==================
``LLMMultiStage``        ✅            ✅            ✅
``TextMatching``         ✅            ❌            ❌
``SAGEE``                ❌            ❌            ✅
======================  ============  ============  ==================

对没有文字标注的空间（走廊、设备间、阳台…）仍然能给出类型，代价是需要训练。

部署形态
--------
纯 numpy，**不需要 torch / DGL**，也不需要子进程 —— 与 ``VecFloorSeg`` 的隔离方式不同。
一份可用的模型是**三个文件**，必须成套使用：

* ``<name>.npz``        权重（``scripts/train_sagee.py --export-weights``）
* ``<name>.labels.json`` 类别表（``--labels-from`` 复制而来）—— **必需**
* ``<name>.norm.json``  逐列 z-score 参数（``--norm-out``）—— **仅当训练时开了
  ``--normalize`` 才需要**；论文忠实配方不标准化，此时该文件不存在

⚠️ 类别表是**硬依赖**：缺失或顺序不一致会让下标整体错位、静默给出错误结果，
所以本类在构造时就要求它存在。标准化参数则按 ``<weights>.meta.json`` 里记录的
配方决定要不要（训练脚本会自动写这个文件并清除不匹配的遗留 ``.norm.json``）。

⚠️ 训练/推理的**轮廓算法必须一致**。SAGE-E 学的是"某种划分下的房间图"，实测
CDT 的过度合并会让一批类别根本不可达（见 ``docs/sagee_baseline_plan.md`` §5.3）。
本模型默认按 **RGP** 轮廓训练，换轮廓算法后需要重新导出并重训。
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence

from ..contracts import ISpaceTypeClassifier
from ..domain import (
    SOURCE_LLM_INFERENCE,
    SpatialComponent,
    SpatialContour,
    SpaceTypePrediction,
    TextAnnotation,
)
from ..registry import SPACE_TYPE_CLASSIFIERS
from ..sagee.graph import from_domain as build_from_domain
from ..sagee.labels import LabelSpace
from ..sagee.model import SageeNumpy, load_weights_meta

__all__ = ['SageeClassifier']


def _sidecar(path: str, suffix: str) -> str:
    return os.path.splitext(path)[0] + suffix


@SPACE_TYPE_CLASSIFIERS.register('SAGEE', aliases=('SAGE_E', 'SAGE', 'SAGEBIM'))
class SageeClassifier(ISpaceTypeClassifier):
    """用 SAGE-E（图注意力网络）预测空间类型。"""

    name = 'SAGEE'

    def __init__(
        self,
        *,
        # ---- 与其它分类器共享的参数（必须存在，否则创建时 TypeError）----
        match_tolerance_mm: float = 0.0,
        min_confidence: float = 0.0,
        # ---- SAGEE 专有参数 ----
        weights: str | None = None,
        norm: str | None = None,
        labels: str | None = None,
        report_confidence: bool = True,
    ):
        #: SAGE-E 不做文字落位，此参数仅为兼容共享配置而接受
        self.match_tolerance_mm = float(match_tolerance_mm or 0.0)
        # 环境变量优先于配置里的共享默认值（settings.yaml 的 params.min_confidence 是
        # 三个分类器共用的），便于在不改配置文件的情况下扫阈值。
        env_conf = os.environ.get('SAGEE_MIN_CONFIDENCE')
        self.min_confidence = float(env_conf) if env_conf not in (None, '') else float(min_confidence or 0.0)
        self.weights = weights or os.environ.get('SAGEE_WEIGHTS', '')
        self.norm = norm or os.environ.get('SAGEE_NORM', '') or None
        self.labels = labels or os.environ.get('SAGEE_LABELS', '') or None
        self.report_confidence = bool(report_confidence)

        self._model: SageeNumpy | None = None
        self._space: LabelSpace | None = None
        self._last_error = ''

    # ------------------------------------------------------------------ 惰性加载
    def _resolve(self) -> tuple[SageeNumpy, LabelSpace]:
        """首次调用时加载权重 / 标准化 / 类别表，并校验三者齐全。"""
        if self._model is not None and self._space is not None:
            return self._model, self._space

        if not self.weights:
            raise RuntimeError(
                '未配置 SAGE-E 权重。请设置 spatial.classification.'
                'algorithm_params.SAGEE.weights 或环境变量 SAGEE_WEIGHTS。')
        if not os.path.isfile(self.weights):
            raise RuntimeError('SAGE-E 权重不存在: %s' % self.weights)

        # 标准化参数是**可选**的："论文忠实"配方（不标准化）下根本没有这个文件，
        # 而标准化配方下它必不可少。区分两种情况的依据是权重旁边的 .meta.json。
        meta = load_weights_meta(self.weights)
        norm_path = self.norm if self.norm else _sidecar(self.weights, '.norm.json')
        has_norm = os.path.isfile(norm_path)
        if self.norm and not has_norm:
            raise RuntimeError('指定的标准化参数不存在: %s' % norm_path)
        if not has_norm:
            norm_path = None
        # 配方与产物不一致时告警（不报错）：套错标准化是静默出错，最难查
        want_norm = meta.get('normalize')
        if want_norm is True and not has_norm:
            raise RuntimeError(
                '权重是用 --normalize 训练的，但找不到标准化参数 %s。\n'
                '缺少它输入分布会完全错位，结果毫无意义。用 SAGEE_NORM 显式指定。'
                % _sidecar(self.weights, '.norm.json'))
        if want_norm is False and has_norm:
            print('[SAGEE] 警告：权重 meta 记录为不标准化，但存在 %s；'
                  '将按 meta 忽略它。' % norm_path)
            norm_path = None

        label_path = self.labels if self.labels else _sidecar(self.weights, '.labels.json')
        if not os.path.isfile(label_path):
            raise RuntimeError(
                '缺少类别表 %s。\n'
                '类别下标 → 类名的对应关系必须与训练时完全一致，因此本文件不可省略。'
                % label_path)

        self._model = SageeNumpy(self.weights, norm=norm_path)
        self._space = LabelSpace.load(label_path)
        return self._model, self._space

    # ------------------------------------------------------------------ API
    def classify(
        self,
        *,
        contours: Sequence[SpatialContour] = (),
        texts: Sequence[TextAnnotation] = (),      # 刻意不用
        components: Sequence[SpatialComponent] = (),
        adjacency: Mapping[str, Sequence[str]] | None = None,
        context: Mapping | None = None,
    ) -> list[SpaceTypePrediction]:
        contours = list(contours)
        if not contours:
            return []

        try:
            model, space = self._resolve()
        except Exception as exc:                                   # noqa: BLE001
            self._last_error = '%s: %s' % (type(exc).__name__, exc)
            print('[SAGEE] %s' % self._last_error)
            return []

        graph = build_from_domain(list(contours), list(components), adjacency,
                                  source='semantic_enricher')
        if graph.n_nodes != len(contours):
            self._last_error = '节点数(%d)与轮廓数(%d)不一致' % (graph.n_nodes, len(contours))
            print('[SAGEE] %s' % self._last_error)
            return []

        proba = model.predict_proba(graph.node_feat, graph.edge_index, graph.edge_feat)

        predictions: list[SpaceTypePrediction] = []
        dropped = 0
        for i, contour_id in enumerate(graph.node_ids):
            idx = int(proba[i].argmax())
            confidence = float(proba[i][idx])
            if confidence < self.min_confidence:
                dropped += 1
                continue
            predictions.append(SpaceTypePrediction(
                contour_id=contour_id,
                types=[space.decode(idx)],
                source=SOURCE_LLM_INFERENCE,
                confidence=confidence if self.report_confidence else None,
            ))

        print('[SAGEE] %d 个空间 -> %d 条预测%s（%d 房间 / %d 边，%d 类）'
              % (graph.n_nodes, len(predictions),
                 '' if not dropped else '（%d 条因置信度 < %.2f 被丢弃）'
                 % (dropped, self.min_confidence),
                 graph.n_nodes, graph.n_edges, space.n_class))
        return predictions

    # ------------------------------------------------------------------ 自检
    @property
    def last_error(self) -> str:
        return self._last_error

    def describe(self) -> dict:
        """给自检脚本用的元信息（不触发加载失败）。"""
        info = {'weights': self.weights, 'name': self.name,
                'min_confidence': self.min_confidence}
        try:
            model, space = self._resolve()
            info.update({
                'n_layer': model.n_layer, 'dim_node': model.ndim_in,
                'dim_edge': model.edim, 'n_class': model.n_class,
                'has_norm': model.norm is not None,
                'class_names': list(space.classes), 'space': space.space,
            })
        except Exception as exc:                                   # noqa: BLE001
            info['error'] = '%s: %s' % (type(exc).__name__, exc)
        return info

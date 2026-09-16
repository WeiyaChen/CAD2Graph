"""``NoOp`` 分类器：**不做任何类型识别**，只把轮廓原样留在图里。

为什么需要它
------------
流水线要求必须有一个分类器。这会带来一个真实的隐患：``--contour-algo CDT``
如果不显式指定 ``--classifier-algo``，就会落到 ``settings.yaml`` 的默认值
（``LLMMultiStage``）—— 于是**为了一次纯轮廓实验白烧 41 次大模型调用**。

它还有第二个作用：让「任务 1 · 空间轮廓提取」这个实验在**概念上干净**。
分类器会**改变轮廓本身** —— 富化阶段有 ACD 复合空间切分，一个空间若被判了多个
类型就会被切开。所以拿 ``TextMatching`` 去跑轮廓实验，量到的并不是提取器的输出。

本分类器返回**空列表**：空间不带任何 ``bldg:`` 类型，也就永远不会被判为复合空间，
更不会被切分。所谓"不动就是最准"。

⚠️ 它**不是**用来"抹平各分类器差异"的 —— 公平比较原则
------------------------------------------------------
ACD 的空间**分割/聚合**是 ``LLMMultiStage`` 的创新点。``SAGE-E`` 没有这个能力，
**不要给它补**任何后处理来模拟 ACD —— 两边都用自己的完整能力，才是公平的比较。
（用户明确要求。）本类的用途仅限于让**轮廓提取**这个实验不带任何语义处理。

ACD 的触发条件见 ``acd_processor._is_composite_space``：一个空间挂了 ≥2 个
``bldg:`` 类型。它与分类器是谁**无关**：``LLMMultiStage`` 会触发，``TextMatching``
命中多处文字时也会触发，``SAGEE`` 是单标签、结构上永不触发。

⚠️ 注意：即便用了 ``NoOp``，评估**任务 1 仍然应该读 ``<base>_raw.jsonld``**
（``topology_builder.build()`` 在富化之前写出的那一份）。那份产物在任何分类器下都
逐位相同，是最无争议的口径 —— 详见 ``docs/sagee_baseline_plan.md``。
"""
from __future__ import annotations

from typing import Mapping, Sequence

from ..contracts import ISpaceTypeClassifier
from ..domain import (
    SpatialComponent,
    SpatialContour,
    SpaceTypePrediction,
    TextAnnotation,
)
from ..registry import SPACE_TYPE_CLASSIFIERS

__all__ = ['NoOpClassifier']


@SPACE_TYPE_CLASSIFIERS.register('NoOp', aliases=('NONE', 'NOCLS', 'CONTOUR_ONLY'))
class NoOpClassifier(ISpaceTypeClassifier):
    """不产出任何类型预测（轮廓专用实验的占位分类器）。"""

    name = 'NoOp'

    def __init__(
        self,
        *,
        # 与其它分类器共享的参数：必须接受，否则工厂合并共享配置时会 TypeError
        match_tolerance_mm: float = 0.0,
        min_confidence: float = 0.0,
        verbose: bool = True,
    ):
        self.match_tolerance_mm = float(match_tolerance_mm or 0.0)
        self.min_confidence = float(min_confidence or 0.0)
        self.verbose = bool(verbose)

    def classify(
        self,
        *,
        contours: Sequence[SpatialContour] = (),
        texts: Sequence[TextAnnotation] = (),
        components: Sequence[SpatialComponent] = (),
        adjacency: Mapping[str, Sequence[str]] | None = None,
        context: Mapping | None = None,
    ) -> list[SpaceTypePrediction]:
        if self.verbose:
            print('[NoOp] %d 个空间：不做类型识别（轮廓专用模式）' % len(contours))
        return []

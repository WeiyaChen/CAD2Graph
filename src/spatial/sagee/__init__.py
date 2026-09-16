# src/spatial/sagee
"""SAGE-E baseline 的纯 numpy 实现与图构造（空间类型识别，无 torch / DGL 依赖）。

模块
----
``model.py``
    SAGE-E 前向的 numpy 端口（4 层、15,394 参数），从 ``.npz`` 权重加载。
``features.py`` / ``graph.py`` / ``labels.py``
    CAD2Graph 空间轮廓 → 8/5 维图特征；本体映射表与长尾折叠。

设计文档见 ``docs/sagee_baseline_plan.md``。
"""

from .model import SageeNumpy, load_weights  # noqa: F401

__all__ = ['SageeNumpy', 'load_weights']

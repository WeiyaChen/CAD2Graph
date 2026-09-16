# src/spatial/contours/__init__.py
"""空间轮廓提取算法集合。

导入本包即完成所有轮廓算法的注册。
"""

from .cdt import CDTContourExtractor
from .filters import SpaceShapeFilter
from .gt import GTContourExtractor
from .rgp import RGPContourExtractor
from .vecfloorseg import VecFloorSegContourExtractor

__all__ = [
    "CDTContourExtractor",
    "GTContourExtractor",
    "RGPContourExtractor",
    "VecFloorSegContourExtractor",
    "SpaceShapeFilter",
]

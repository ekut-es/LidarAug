from enum import Enum
from typing import Optional
from torch import Tensor


class FogMetric(Enum):
    DIST: int
    CHAMFER: int


def fog(point_cloud: Tensor, prob: float, metric: FogMetric, sigma: float,
        mean: int) -> Optional[list[Tensor]]:
    ...

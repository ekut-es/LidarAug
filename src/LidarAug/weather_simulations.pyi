from enum import Enum
from typing import Optional
from torch import Tensor


class FogMetric(Enum):
    """
    Different parameters for the fog model/simulation.

    DIST: Optimization of the distance distribution between the points.
    CHAMFER: Optimization of the chamfer distance.
    """
    DIST: int
    CHAMFER: int


def fog(point_cloud: Tensor, prob: float, metric: FogMetric, sigma: float,
        mean: int) -> Optional[list[Tensor]]:
    ...

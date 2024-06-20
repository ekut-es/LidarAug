from enum import Enum
from typing import Optional, overload
from torch import Tensor


class FogParameter(Enum):
    """
    Different parameters for the fog model/simulation.

    DIST: Optimization of the distance distribution between the points.
    CHAMFER: Optimization of the chamfer distance.
    """
    DIST: int
    CHAMFER: int


@overload
def fog(point_cloud: Tensor, prob: float, metric: FogParameter, sigma: float,
        mean: int) -> Optional[list[Tensor]]:
    ...


@overload
def fog(point_cloud: Tensor, metric: FogParameter, viewing_dist: float,
        max_intensity: int) -> Tensor:
    ...

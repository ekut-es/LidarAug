from enum import Enum
from typing import Optional, overload
from torch import Tensor
from point_cloud import IntensityRange


class FogParameter(Enum):
    """
    Different parameters for the fog model/simulation.

    DIST: Optimization of the distance distribution between the points.
    CHAMFER: Optimization of the chamfer distance.
    """
    DIST: int
    CHAMFER: int


class Distribution(Enum):
    """
    Different options to determine which statistical distribution should
    be used to sample the particles for some weather simulations.
    """
    EXPONENTIAL: int
    LOG_NORMAL: int
    GM: int


@overload
def fog(point_cloud: Tensor, prob: float, metric: FogParameter, sigma: float,
        mean: int) -> Optional[list[Tensor]]:
    ...


@overload
def fog(point_cloud: Tensor, metric: FogParameter, viewing_dist: float,
        max_intensity: IntensityRange) -> Tensor:
    ...


def rain(point_cloud: Tensor, dims: list[float], num_drops: int,
         precipitation: float, d: Distribution,
         max_intensity: IntensityRange) -> Tensor:
    ...


def snow(point_cloud: Tensor, dims: list[float], num_drops: int,
         precipitation: float, scale: int, max_intensity: IntensityRange):
    ...

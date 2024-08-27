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
    """
    Applies a rain simulation to a point cloud.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param dims: set the upper and lower bounds of the uniform distribution used to draw new points for the noise filter.
    :param num_drops: is the number of rain drops per m^3.
    :param precipitation: is the precipitation rate and determines the raindrop size distribution.
    :param d: is the distribution function used when sampling the particles.
    :param max_intensity: is the maximum intensity of the points in the point cloud.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


def snow(point_cloud: Tensor, dims: list[float], num_drops: int,
         precipitation: float, scale: int,
         max_intensity: IntensityRange) -> Tensor:
    ...

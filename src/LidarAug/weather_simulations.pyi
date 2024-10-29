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
    """
    Applies a fog simulation to a point cloud with a chance of `prob`%.
    The point cloud has the shape (B, P, F).

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param prob: is the probability with which the simulation is applied.
    :param metric: is a parameter used to control the simulation.
    :param sigma: is the standard deviation used to draw the viewing distance in the fog.
    :param mean: is the mean that is used to draw the viewing distance in the fog.
    :return: A list of B tensors with P points that the simulation has been applied to or None.
    """
    ...


@overload
def fog(point_cloud: Tensor, metric: FogParameter, viewing_dist: float,
        max_intensity: IntensityRange) -> Tensor:
    """
    Applies a fog simulation to a point cloud.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param metric: is a parameter used to control the simulation.
    :param viewing_dist:  is the viewing distance in the fog.
    :param max_intensity:  is the maximum intensity value of a point.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@overload
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


@overload
def rain(point_cloud: Tensor, noise_filter_path: str, num_drops_sigma: int,
         precipitation_sigma: float, prob: float) -> Optional[Tensor]:
    """
    Applies a rain simulation to a point cloud with a chance of `prob`%.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param noise_filter_path: is the path to the directory containing the npz files with the noise filter data.
    :param num_drops_sigma: is the standard deviation for the number of drops (used to find the correct noise filter).
    :param precipitation_sigma: is the standard deviation for the precipitation rate (used to find the correct noise filter).
    :param prob: is the probability that the simulation will be executed.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@overload
def snow(point_cloud: Tensor, dims: list[float], num_drops: int,
         precipitation: float, scale: int,
         max_intensity: IntensityRange) -> Tensor:
    """
    Applies a snow simulation to a point cloud.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param dims: set the upper and lower bounds of the uniform distribution used to draw new points for the noise filter.
    :param num_drops: is the number of snow flakes per m^3.
    :param precipitation: is the precipitation rate and determines the snowflake size distribution.
    :param scale: is used to scale the size of the sampled particles when generating the noise filter.
    :param max_intensity: is the maximum intensity of the points in the point cloud.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@overload
def snow(point_cloud: Tensor, noise_filter_path: str, num_drops_sigma: int,
         precipitation_sigma: float, scale: int,
         prob: float) -> Optional[Tensor]:
    """
    Applies a snow simulation to a point cloud with a chance of `prob`%.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param noise_filter_path: is the path to the directory containing the npz files with the noise filter data.
    :param num_drops_sigma: is the standard deviation for the number of snow flakes (used to find the correct noise filter).
    :param precipitation_sigma: is the standard deviation for the precipitation rate (used to find the correct noise filter).
    :param scale: is used to scale the size of the sampled particles when generating the noise filter.
    :param prob: is the probability that the simulation will be executed.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...

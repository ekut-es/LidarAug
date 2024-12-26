from enum import Enum
import lidar_aug.point_cloud
import lidar_aug.transformations
import torch
import typing

__all__ = ['Distribution', 'FogParameter', 'fog', 'rain', 'snow']


class Distribution(Enum):
    """
    Different options to determine which statistical distribution shouldbe used to sample the particles for some weather simulations.

    Members:

      EXPONENTIAL : Exponential distribution.

      LOG_NORMAL : Log normal distribution.

      GM : GM distribution.
    """
    EXPONENTIAL: typing.ClassVar[
        Distribution]  # value = <Distribution.EXPONENTIAL: 0>
    GM: typing.ClassVar[Distribution]  # value = <Distribution.GM: 2>
    LOG_NORMAL: typing.ClassVar[
        Distribution]  # value = <Distribution.LOG_NORMAL: 1>
    __members__: typing.ClassVar[dict[
        str,
        Distribution]]  # value = {'EXPONENTIAL': <Distribution.EXPONENTIAL: 0>, 'LOG_NORMAL': <Distribution.LOG_NORMAL: 1>, 'GM': <Distribution.GM: 2>}

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: int) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


class FogParameter(Enum):
    """
    Different parameters for the fog model/simulation.

    Members:

      DIST : Optimization of the distance distribution between the points.

      CHAMFER : Optimization of the chamfer distance.
    """
    CHAMFER: typing.ClassVar[FogParameter]  # value = <FogParameter.CHAMFER: 1>
    DIST: typing.ClassVar[FogParameter]  # value = <FogParameter.DIST: 0>
    __members__: typing.ClassVar[dict[
        str,
        FogParameter]]  # value = {'DIST': <FogParameter.DIST: 0>, 'CHAMFER': <FogParameter.CHAMFER: 1>}

    def __eq__(self, other: typing.Any) -> bool:
        ...

    def __getstate__(self) -> int:
        ...

    def __hash__(self) -> int:
        ...

    def __index__(self) -> int:
        ...

    def __init__(self, value: int) -> None:
        ...

    def __int__(self) -> int:
        ...

    def __ne__(self, other: typing.Any) -> bool:
        ...

    def __repr__(self) -> str:
        ...

    def __setstate__(self, state: int) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def value(self) -> int:
        ...


@typing.overload
def fog(point_cloud: torch.Tensor, prob: float, metric: FogParameter,
        sigma: float, mean: int) -> typing.Optional[list[torch.Tensor]]:
    """
    Applies a fog simulation to a point cloud with a chance of `prob` %.
    The point cloud has the shape `(B, N, F)` where `B` is the number of batches, `N` is the number of points and `F` is the number of features, which is 4; `(x, y, z, i)`.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param prob: is the probability with which the simulation is applied.
    :param metric: is a parameter used to control the simulation.
    :param sigma: is the standard deviation used to draw the viewing distance in the fog.
    :param mean: is the mean that is used to draw the viewing distance in the fog.
    :return: A list of B tensors with P points that the simulation has been applied to or None.
    """
    ...


@typing.overload
def fog(
    point_cloud: torch.Tensor,
    metric: FogParameter,
    viewing_dist: float,
    max_intensity: lidar_aug.point_cloud.IntensityRange = lidar_aug.
    point_cloud.IntensityRange.MAX_INTENSITY_1
) -> torch.Tensor:
    """
    Applies a fog simulation to a point cloud.
    The point cloud has the shape `(N, F)` where `N` is the number of points and `F` is the number of features, which is 4; `(x, y, z, i)`.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param metric: is a parameter used to control the simulation.
    :param viewing_dist:  is the viewing distance in the fog.
    :param max_intensity:  is the maximum intensity value of a point.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@typing.overload
def rain(
    point_cloud: torch.Tensor,
    dims: lidar_aug.transformations.DistributionRanges,
    num_drops: int,
    precipitation: float,
    d: Distribution,
    max_intensity: lidar_aug.point_cloud.IntensityRange = lidar_aug.
    point_cloud.IntensityRange.MAX_INTENSITY_1
) -> torch.Tensor:
    """
    Applies a rain simulation to a point cloud.
    The point cloud has the shape `(N, F)` where `N` is the number of points and `F` is the number of features, which is 4; `(x, y, z, i)`.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param dims: set the upper and lower bounds of the uniform distribution used to draw new points for the noise filter.
    :param num_drops: is the number of rain drops per m^3.
    :param precipitation: is the precipitation rate and determines the raindrop size distribution.
    :param d: is the distribution function used when sampling the particles.
    :param max_intensity: is the maximum intensity of the points in the point cloud.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@typing.overload
def rain(point_cloud: torch.Tensor, noise_filter_path: str,
         num_drops_sigma: int, precipitation_sigma: float,
         prob: float) -> typing.Optional[torch.Tensor]:
    """
    Applies a rain simulation to a point cloud with a chance of `prob` %.
    The point cloud has the shape `(N, F)` where `N` is the number of points and `F` is the number of features, which is 4; `(x, y, z, i)`.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param noise_filter_path: is the path to the directory containing the npz files with the noise filter data.
    :param num_drops_sigma: is the standard deviation for the number of drops (used to find the correct noise filter).
    :param precipitation_sigma: is the standard deviation for the precipitation rate (used to find the correct noise filter).
    :param prob: is the probability that the simulation will be executed.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@typing.overload
def snow(
    point_cloud: torch.Tensor,
    dims: lidar_aug.transformations.DistributionRanges,
    num_drops: int,
    precipitation: float,
    scale: int,
    max_intensity: lidar_aug.point_cloud.IntensityRange = lidar_aug.
    point_cloud.IntensityRange.MAX_INTENSITY_1
) -> torch.Tensor:
    """
    Applies a snow simulation to a point cloud.
    The point cloud has the shape `(N, F)` where `N` is the number of points and `F` is the number of features, which is 4; `(x, y, z, i)`.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param dims: set the upper and lower bounds of the uniform distribution used to draw new points for the noise filter.
    :param num_drops: is the number of snowflakes per m^3.
    :param precipitation: is the precipitation rate and determines the snowflake size distribution.
    :param scale: is used to scale the size of the sampled particles when generating the noise filter.
    :param max_intensity: is the maximum intensity of the points in the point cloud.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...


@typing.overload
def snow(point_cloud: torch.Tensor, noise_filter_path: str,
         num_drops_sigma: int, precipitation_sigma: float, scale: int,
         prob: float) -> typing.Optional[torch.Tensor]:
    """
    Applies a snow simulation to a point cloud with a chance of `prob` %.
    The point cloud has the shape `(N, F)` where `N` is the number of points and `F` is the number of features, which is 4; `(x, y, z, i)`.

    :param point_cloud: is the point cloud that the simulation is applied to.
    :param noise_filter_path: is the path to the directory containing the npz files with the noise filter data.
    :param num_drops_sigma: is the standard deviation for the number of snowflakes (used to find the correct noise filter).
    :param precipitation_sigma: is the standard deviation for the precipitation rate (used to find the correct noise filter).
    :param scale: is used to scale the size of the sampled particles when generating the noise filter.
    :param prob: is the probability that the simulation will be executed.
    :return: a new point cloud with the old points as a base but after applying the simulation.
    """
    ...

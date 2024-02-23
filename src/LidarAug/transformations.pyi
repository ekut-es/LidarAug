from enum import Enum
from torch import Tensor


class NoiseType(Enum):
    UNIFORM: int
    SALT_PEPPER: int
    MIN: int
    MAX: int


class ItensityRange(Enum):
    """
    Defines options for maximum intensity values.
    Intensity goes from [0; MAX_INTENSITY], where MAX_INTENSITY is either 1 or
    255.
    """
    MAX_INTENSITY_1: int
    MAX_INTENSITY_255: int


def translate(points: Tensor, translation: Tensor) -> None:
    """
    Moves points by a specific amount.

    :param points:      is the point cloud with the points are to be moved.
    :param translation: is the translation vector that specifies by how much they points are moved.
    """
    ...


def translate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    """
    Generates a random (3D) translation vector using a normal distribution and applies it to all the points and labels.

    :param points: is the point cloud with the points that are translated.
    :param labels: are the labels belonging to the aforementioned point cloud.
    :param sigma:  is the standard deviation of the normal distribution.
    """
    ...


def scale_points(points: Tensor, scaling_factor: float) -> None:
    """
    Scales points by a constant factor.
    Point cloud is expected to be of shape (b, n, f), where `b` is the number of batches, `n` is the number of points and `f` is the number of features.
    `f` is supposed to be 4.

    :param points:         is the point cloud whose points are scaled.
    :param scaling_factor: is the factor that the (x, y, z) coordinates are multiplied by.
    """
    ...


def scale_random(points: Tensor, labels: Tensor, sigma: float,
                 max_scale: float) -> None:
    """
    Scales the points and labels by a random factor.
    This factor is drawn from a truncated normal distribution.
    The truncated normal distribution has a mean of 1. The standard deviation, as
    well as upper and lower limits are determined by the function parameters.

    :param points:    is the point cloud that contains the points that will be scaled.
    :param labels:    are the labels belonging to the aforementioned point cloud.
    :param sigma:     is the the standard deviation of the truncated normal distribution.
    :param max_scale: is the upper limit of the truncated normal distribution. The lower limit is the inverse.
    """
    ...


def scale_local(points: Tensor, labels: Tensor, sigma: float,
                max_scale: float) -> None:
    """
    Scales the points that are part of a box and the corresponding labels by a
    random factor.

    This factor is drawn from a truncated normal distribution.
    The truncated normal distribution has a mean of 1. The standard deviation, as
    well as upper and lower limits are determined by the function parameters.

    :param points:    is the point cloud that contains the points that will be scaled.
    :param labels:    are the labels belonging to the aforementioned point cloud.
    :param sigma:     is the the standard deviation of the truncated normal distribution.
    :param max_scale: is the upper limit of the truncated normal distribution. The lower limit is the inverse.
    """
    ...


def flip_random(points: Tensor, labels: Tensor, prob: int) -> None:
    """
    Flips all the points in the point cloud with a probability of `prob`% in the direction of the y-axis.

    :param points:  is the point cloud containing the points that will be flipped.
    :param labels:  are the corresponding labels.
    :param prob:    is the probability with which the points should be flipped.
    """
    ...

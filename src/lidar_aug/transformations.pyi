from enum import Enum
from typing import Tuple, overload
from torch import Tensor

from lidar_aug.point_cloud import IntensityRange


class NoiseType(Enum):
    """
    Indicates how the noise is added:

    UNIFORM: The noise values are drawn from a uniform distribution.
    SALT_PEPPER: Half of the added values have the maximum intensity, the other half the minimum intensity.
    MIN: The noise values are equal to the minimum intensity.
    MAX: The noise values are equal to the maximum intensity.
    """
    UNIFORM: int
    SALT_PEPPER: int
    MIN: int
    MAX: int


class DistributionRange:
    min: float
    max: float

    def __init__(self, min: float, max: float) -> None:
        ...


class DistributionRanges:
    x_range: DistributionRange
    y_range: DistributionRange
    z_range: DistributionRange
    uniform_range: DistributionRange

    @overload
    def __init__(self, x_range: DistributionRange, y_range: DistributionRange,
                 z_range: DistributionRange,
                 uniform_range: DistributionRange) -> None:
        ...

    @overload
    def __init__(self, x_range: DistributionRange, y_range: DistributionRange,
                 z_range: DistributionRange) -> None:
        ...


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
    :param sigma:     is the standard deviation of the truncated normal distribution.
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
    :param sigma:     is the standard deviation of the truncated normal distribution.
    :param max_scale: is the upper limit of the truncated normal distribution. The lower limit is the inverse.
    """
    ...


def flip_random(points: Tensor, labels: Tensor, prob: int) -> None:
    """
    Flips all the points in the point cloud with a probability of `prob` % in the direction of the y-axis.

    :param points:  is the point cloud containing the points that will be flipped.
    :param labels:  are the corresponding labels.
    :param prob:    is the probability with which the points should be flipped.
    """
    ...


def random_noise(points: Tensor, sigma: float,
                 ranges: list[float] | DistributionRanges,
                 noise_type: NoiseType,
                 max_intensity: IntensityRange) -> Tensor:
    """
    Adds random amount of points (drawn using a normal distribution) at random coordinates
    (within predetermined ranges) with a random intensity according to specific noise type.

    :param points:         is the point cloud that the points will be added to.
    :param sigma:          is the standard deviation of the normal distribution that is used to draw the number of points to be added.
    :param ranges:         are the boundaries in (min and max (x, y, z) values) in which the new points can be created.
    :param noise_type:     is one of a number of 'patterns' that can be used to generate the points.
    :param max_intensity:  is the maximum intensity value in the dataset.
    """
    ...


def thin_out(points: Tensor, sigma: float) -> Tensor:
    """
    Randomly generates a percentage from a norma distribution, which determines
    how many items should be 'thinned out'. From that percentage random indices
    are uniformly drawn (in a random order, where each index is unique).

    Finally, a new tensor is created containing the items present at those
    indices.

    :param points: is the point cloud.
    :param sigma:  is the standard deviation of the distribution that generates the percentage.
    :return: a new tensor containing the new set of points.
    """
    ...


def rotate_deg(points: Tensor, angle: float) -> None:
    """
    Rotates a batch of points along the 'z' axis (yaw).

    :param points: is the point cloud that the rotation is applied to.
    :param angle:  is the angle (in degrees) by which the points are to be rotated.
    """
    ...


def rotate_rad(points: Tensor, angle: float) -> None:
    """
    Rotates a batch of points along the 'z' axis (yaw).

    :param points: is the point cloud that the rotation is applied to.
    :param angle:  is the angle (in radians) by which the points are to be rotated.
    """
    ...


def rotate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    """
    Rotates points and labels.
    The number of degrees that they are rotated by is determined by a randomly generated value from a normal distribution.

    :param points: is the point cloud that the rotation is applied to.
    :param labels: are the labels belonging to the point cloud that the rotation is applied to.
    :param sigma:  is the standard deviation of the normal distribution.
    """
    ...


def delete_labels_by_min_points(points: Tensor, labels: Tensor, names: Tensor,
                                min_points: int) -> Tuple[Tensor, Tensor]:
    """
    Checks the amount of points for each bounding box.
    If the number of points is smaller than a given threshold, the box is removed
    along with its label.

    :param points:     is the point_cloud.
    :param labels:     are the bounding boxes of objects.
    :param names:      are the names/labels of these boxes.
    :param min_points: is the point threshold.
    :return: The batch with the new labels and the batch with the new names.
    """
    ...


def random_point_noise(points: Tensor, sigma: float) -> None:
    """
    Moves each point in the point cloud randomly.
    How much each coordinate is changed is decided by values drawn from a normal distribution.

    :param points: is the point cloud from which each point is moved.
    :param sigma:  is the standard deviation of the normal distribution.
    """
    ...


def transform_along_ray(points: Tensor, sigma: float) -> None:
    """
    Moves each point in the point cloud randomly along a ray.
    How much it is moved is decided by a value drawn from a normal distribution.

    :param points: is the point cloud from which each point is moved.
    :param sigma:  is the standard deviation of the normal distribution.
    """
    ...


def intensity_noise(points: Tensor, sigma: float,
                    max_intensity: IntensityRange) -> None:
    """
    Shifts the intensity value of every point in the point cloud by a random amount drawn from a normal distribution.

    :param points:        is the point cloud with all the points.
    :param sigma:         is the standard deviation of the normal distribution.
    :param max_intensity: is the maximum intensity value (either 1 or 255, depending on the dataset).
    """
    ...


def intensity_shift(points: Tensor, sigma: float,
                    max_intensity: IntensityRange) -> None:
    """
    Shifts the intensity value of every point in the point cloud by a single value drawn from a normal distribution.

    :param points:        is the point cloud with all the points.
    :param sigma:         is the standard deviation of the normal distribution.
    :param max_intensity: is the maximum intensity value (either 1 or 255, depending on the dataset).
    """
    ...


def local_to_local_transform(from_pose: Tensor, to_pose: Tensor) -> Tensor:
    """
    Creates a transformation matrix from the local system into a 'target' coordinate frame.

    :param from_pose: is the local coordinate frame (x, y, z, roll, yaw, pitch).
    :param to_pose:   is the target coordinate frame (x, y, z, roll, yaw, pitch).
    :return: the homogeneous transformation matrix into the target coordinate frame.
    """
    ...


def local_to_world_transform(lidar_pose: Tensor) -> Tensor:
    """
    Creates a transformation matrix from the local system into the global coordinate frame.

    :param lidar_pose: is the local coordinate frame (x, y, z, roll, yaw, pitch).
    :return: the homogeneous transformation matrix into the global coordinate frame.
    """
    ...


def apply_transformation(points: Tensor,
                         transformation_matrix: Tensor) -> None:
    """
    Applies a transformation matrix to an entire point cloud with the shape (B,
    N, F), where B is the number of batches and N is the number of points.

    :param points:                is the point cloud that the transformation
                                  matrix is applied to.
    :param transformation_matrix: is the transformation matrix.
    """
    ...


def change_sparse_representation(input: Tensor, batch_idx: int) -> Tensor:
    """
    Changes the representation of a sparse tensor from a flat 2D tensor (N, F),
    where F is the number of features to a 3D tensor (B, n, f), where B is the
    number of batches, n is the number of tensors in each batch and f is the
    number of features (equal to F-1).
    0s are used for padding.

    :param input:     is the input tensor.
    :param batch_idx: is the index of the batch index.

    :return: a new tensor with 0s for padding.
    """
    ...

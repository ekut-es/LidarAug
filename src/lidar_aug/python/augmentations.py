from torch import Tensor
from lidar_aug import transformations
from lidar_aug.point_cloud import IntensityRange


def _check_points(points: Tensor) -> None:
    """
    Performs a bunch of assertions to make sure that a point cloud has the right shape.
    """

    shape = points.shape
    assert len(
        shape
    ) == 3, "Tensor is not of shape (B, N, F), where B is the batch-size, N is the number of points and F is the number of features!"
    assert shape[
        2] == 4, "point is supposed to have 4 components (x, y, z, intensity)!"


def _check_labels(labels: Tensor) -> None:
    """
    Performs a bunch of assertions to make sure that a set of labels has the right shape.
    """

    shape = labels.shape
    assert len(
        shape
    ) == 3, "Tensor is not of shape (B, N, F), where B is the batch-size, N is the number of labels and F is the number of features!"
    assert shape[
        2] == 7, "label is supposed to have 7 components (x, y, z, width, height, length, theta)!"


def _check_labels_and_points(points: Tensor, labels: Tensor) -> None:
    """
    Performs a bunch of assertions to make sure that a point cloud and the corresponding labels have the right shapes.
    """

    shape_points = points.shape
    shape_labels = labels.shape

    assert shape_points[0] == shape_labels[
        0], "Batch sizes for points and labels are not equal!"
    _check_points(points)
    _check_labels(labels)


def _check_frame_coordinate_dimensions(frame: Tensor) -> None:
    """
    Checks whether a frame has the correct shape.
    """
    shape = frame.shape
    assert len(shape) == 1 and shape[
        0] == 6, "`frame` is supposed to be a 6-vector (x, y, z, roll, yaw, pitch)"


def translate(points: Tensor, translation: Tensor) -> None:
    """
    Moves points by a specific amount.

    :param points:      is the point cloud with the points are to be moved. Expected shape is `(B, N, 4)`.
    :param translation: is the translation vector that specifies by how much they points are moved.
    """

    _check_points(points)

    transformations.translate(points, translation)


def translate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    """
    Generates a random (3D) translation vector using a normal distribution and applies it to all the points and labels.

    :param points: is the point cloud with the points that are translated. Expected shape is `(B, N, 4)`.
    :param labels: are the labels belonging to the aforementioned point cloud. Expected shape is `(B, N, 7)`.
    :param sigma:  is the standard deviation of the normal distribution.
    """

    _check_labels_and_points(points, labels)

    transformations.translate_random(points, labels, sigma)


def scale(points: Tensor, scaling_factor: float) -> None:
    """
    Scales points by a constant factor.

    :param points:         is the point cloud whose points are scaled. Expected shape is `(B, N, 4)`.
    :param scaling_factor: is the factor that the (x, y, z) coordinates are multiplied by.
    """

    _check_points(points)

    transformations.scale_points(points, scaling_factor)


def scale_random(points: Tensor, labels: Tensor, sigma: float,
                 max_scale: float) -> None:
    """
    Scales the points and labels by a random factor.
    This factor is drawn from a truncated normal distribution.
    The truncated normal distribution has a mean of 1. The standard deviation, as
    well as upper and lower limits are determined by the function parameters.

    :param points:    is the point cloud that contains the points that will be scaled. Expected shape is `(B, N, 4)`.
    :param labels:    are the labels belonging to the aforementioned point cloud. Expected shape is `(B, N, 7)`.
    :param sigma:     is the standard deviation of the truncated normal distribution.
    :param max_scale: is the upper limit of the truncated normal distribution. The lower limit is the inverse.
    """

    _check_labels_and_points(points, labels)

    transformations.scale_random(points, labels, sigma, max_scale)


def scale_local(points: Tensor, labels: Tensor, sigma: float,
                max_scale: float) -> None:
    """
    Scales the points that are part of a box and the corresponding labels by a
    random factor.

    This factor is drawn from a truncated normal distribution.
    The truncated normal distribution has a mean of 1. The standard deviation, as
    well as upper and lower limits are determined by the function parameters.

    :param points:    is the point cloud that contains the points that will be scaled. Expected shape is `(B, N, 4)`.
    :param labels:    are the labels belonging to the aforementioned point cloud. Expected shape is `(B, N, 7)`.
    :param sigma:     is the standard deviation of the truncated normal distribution.
    :param max_scale: is the upper limit of the truncated normal distribution. The lower limit is the inverse.
    """

    _check_labels_and_points(points, labels)

    transformations.scale_local(points, labels, sigma, max_scale)


def flip_random(points: Tensor, labels: Tensor, prob: int) -> None:
    """
    Flips all the points in the point cloud with a probability of `prob` % in the direction of the y-axis.

    :param points:  is the point cloud containing the points that will be flipped. Expected shape is `(B, N, 4)`.
    :param labels:  are the corresponding labels. Expected shape is `(B, N, 7)`.
    :param prob:    is the probability with which the points should be flipped.
    """

    assert 0 <= prob <= 100, f"{prob}% is not a valid probability"
    _check_labels_and_points(points, labels)

    transformations.flip_random(points, labels, prob)


def random_noise(points: Tensor, sigma: float,
                 ranges: list[float] | transformations.DistributionRanges,
                 noise_type: transformations.NoiseType,
                 max_intensity: IntensityRange) -> None:
    """
    Adds random amount of points (drawn using a normal distribution) at random coordinates
    (within predetermined ranges) with a random intensity according to specific noise type.

    :param points:         is the point cloud that the points will be added to. Expected shape is `(B, N, 4)`.
    :param sigma:          is the standard deviation of the normal distribution that is used to draw the number of points to be added.
    :param ranges:         are the boundaries in (min and max (x, y, z) values) in which the new points can be created.
    :param noise_type:     is one of a number of 'patterns' that can be used to generate the points.
    :param max_intensity:  is the maximum intensity value in the dataset
    """

    _check_points(points)

    if type(ranges) is list:
        x_min, x_max = ranges[0], ranges[1]
        y_min, y_max = ranges[2], ranges[3]
        z_min, z_max = ranges[4], ranges[5]
        uniform_min, uniform_max = ranges[6], ranges[7]

        distribution_ranges = transformations.DistributionRanges(
            transformations.DistributionRange(x_min, x_max),
            transformations.DistributionRange(y_min, y_max),
            transformations.DistributionRange(z_min, z_max),
            transformations.DistributionRange(uniform_min, uniform_max))
    else:
        distribution_ranges = ranges

    point_cloud = transformations.random_noise(points, sigma,
                                               distribution_ranges, noise_type,
                                               max_intensity)

    points.resize_(point_cloud.shape)
    points.copy_(point_cloud)


def thin_out(points: Tensor, sigma: float) -> None:
    """
    Randomly generates a percentage from a normal distribution, which determines
    how many items should be 'thinned out'. From that percentage random indices
    are uniformly drawn (in a random order, where each index is unique).

    Finally, a new tensor is created containing the items present at those
    indices.

    :param points: is the point cloud. Expected shape is `(B, N, 4)`.
    :param sigma:  is the standard deviation of the distribution that generates the percentage.
    """

    _check_points(points)

    batch_points = transformations.thin_out(points, sigma)
    points.resize_(batch_points.shape)
    points.copy_(batch_points)


def rotate_deg(points: Tensor, angle: float) -> None:
    """
    Rotates a batch of points along the 'z' axis (yaw).

    :param points: is the point cloud that the rotation is applied to. Expected shape is `(B, N, 4)`.
    :param angle:  is the angle (in degrees) by which the points are to be rotated.
    """

    _check_points(points)

    transformations.rotate_deg(points, angle)


def rotate_rad(points: Tensor, angle: float) -> None:
    """
    Rotates a batch of points along the 'z' axis (yaw).

    :param points: is the point cloud that the rotation is applied to. Expected shape is `(B, N, 4)`.
    :param angle:  is the angle (in radians) by which the points are to be rotated.
    """

    _check_points(points)

    transformations.rotate_rad(points, angle)


def rotate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    """
    Rotates points and labels.
    The number of degrees that they are rotated by is determined by a randomly generated value from a normal distribution.

    :param points: is the point cloud that the rotation is applied to. Expected shape is `(B, N, 4)`.
    :param labels: are the labels belonging to the point cloud that the rotation is applied to. Expected shape is `(B, N, 7)`.
    :param sigma:  is the standard deviation of the normal distribution.
    """

    _check_labels_and_points(points, labels)

    transformations.rotate_random(points, labels, sigma)


def delete_labels_by_min_points(points: Tensor, labels: Tensor, names: Tensor,
                                min_points: int) -> None:
    """
     Checks the amount of points for each bounding box.
     If the number of points is smaller than a given threshold, the box is removed
     along with its label.

    :param points:     is the point_cloud. Expected shape is `(B, N, 4)`.
    :param labels:     are the bounding boxes of objects. Expected shape is `(B, N, 7)`.
    :param names:      are the names/labels of these boxes.
    :param min_points: is the point threshold.
    """

    _check_labels_and_points(points, labels)

    batch_labels, batch_names = transformations.delete_labels_by_min_points(
        points, labels, names, min_points)

    labels.resize_(batch_labels.shape)
    labels.copy_(batch_labels)

    names.resize_(batch_names.shape)
    names.copy_(batch_names)


def random_point_noise(points: Tensor, sigma: float) -> None:
    """
    Moves each point in the point cloud randomly.
    How much each coordinate is changed is decided by values drawn from a normal distribution.

    :param points: is the point cloud from which each point is moved. Expected shape is `(B, N, 4)`.
    :param sigma:  is the standard deviation of the normal distribution.
    """

    _check_points(points)

    transformations.random_point_noise(points, sigma)


def transform_along_ray(points: Tensor, sigma: float) -> None:
    """
    Moves each point in the point cloud randomly along a ray.
    How much it is moved is decided by a value drawn from a normal distribution.

    :param points: is the point cloud from which each point is moved. Expected shape is `(B, N, 4)`.
    :param sigma:  is the standard deviation of the normal distribution.
    """

    _check_points(points)

    transformations.transform_along_ray(points, sigma)


def intensity_noise(points: Tensor, sigma: float,
                    max_intensity: IntensityRange) -> None:
    """
    Shifts the intensity value of every point in the point cloud by a random amount drawn from a normal distribution.

    :param points:        is the point cloud with all the points. Expected shape is `(B, N, 4)`.
    :param sigma:         is the standard deviation of the normal distribution.
    :param max_intensity: is the maximum intensity value (either 1 or 255, depending on the dataset).
    """

    _check_points(points)

    transformations.intensity_noise(points, sigma, max_intensity)


def intensity_shift(points: Tensor, sigma: float,
                    max_intensity: IntensityRange) -> None:
    """
    Shifts the intensity value of every point in the point cloud by a single value drawn from a normal distribution.

    :param points:        is the point cloud with all the points. Expected shape is `(B, N, 4)`.
    :param sigma:         is the standard deviation of the normal distribution.
    :param max_intensity: is the maximum intensity value (either 1 or 255, depending on the dataset).
    """

    _check_points(points)

    transformations.intensity_shift(points, sigma, max_intensity)


def local_to_world_transform(lidar_pose: Tensor) -> Tensor:
    """
    Creates a transformation matrix from the local system into the global coordinate frame.

    :param lidar_pose: is the local coordinate frame `(x, y, z, roll, yaw, pitch)`.
    :return: the homogeneous transformation matrix into the global coordinate frame.
    """

    _check_frame_coordinate_dimensions(lidar_pose)

    return transformations.local_to_world_transform(lidar_pose)


def local_to_local_transform(from_pose: Tensor, to_pose: Tensor) -> Tensor:
    """
    Creates a transformation matrix from the local system into a 'target' coordinate frame.

    :param from_pose: is the local coordinate frame `(x, y, z, roll, yaw, pitch)`.
    :param to_pose:   is the target coordinate frame `(x, y, z, roll, yaw, pitch)`.
    :return: the homogeneous transformation matrix into the target coordinate frame.
    """

    _check_frame_coordinate_dimensions(from_pose)
    _check_frame_coordinate_dimensions(to_pose)

    return transformations.local_to_local_transform(from_pose, to_pose)


def apply_transformation(points: Tensor,
                         transformation_matrix: Tensor) -> None:
    """
    Applies a transformation matrix to an entire point cloud with the shape (B,
    N, F), where B is the number of batches and N is the number of points.

    :param points:                is the point cloud that the transformation
                                  matrix is applied to. Expected shape is `(B, N, 4)`.
    :param transformation_matrix: is the transformation matrix.
    """

    _check_points(points)

    transformations.apply_transformation(points, transformation_matrix)

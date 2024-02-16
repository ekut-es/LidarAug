from torch import Tensor
from LidarAug import transformations


def _check_points(points: Tensor) -> None:
    """
    Performs a bunch of assertions to make sure that a point cloud has the right shape.
    """

    shape = points.shape
    assert len(
        shape
    ) == 3, "Tensor is not of shape (B, N, F), where B is the batchsize, N is the number of points and F is the number of features!"
    assert shape[
        2] == 4, "point is supposed to have 4 components (x, y, z, intensity)!"


def _check_labels(labels: Tensor) -> None:
    """
    Performs a bunch of assertions to make sure that a set of labels has the right shape.
    """

    shape = labels.shape
    assert len(
        shape
    ) == 3, "Tensor is not of shape (B, N, F), where B is the batchsize, N is the number of labels and F is the number of features!"
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


def translate(points: Tensor, translation: Tensor) -> None:
    """
    Moves points by a specific amount.

    :param points:      is the point cloud with the points are to be moved.
    :param translation: is the translation vector that specifies by how much they points are moved.
    """

    _check_points(points)

    transformations.translate(points, translation)


def translate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    """
    Generates a random (3D) translation vector using a normal distribution and applies it to all the points and labels.

    :param points: is the point cloud with the points that are translated.
    :param labels: are the labels belonging to the aforementioned point cloud.
    :param sigma:  is the standard deviation of the normal distribution.
    """

    _check_labels_and_points(points, labels)

    transformations.translate_random(points, labels, sigma)


def scale(points: Tensor, scaling_factor: float) -> None:
    """
    Scales points by a constant factor.
    Point cloud is expected to be of shape (b, n, f), where `b` is the number of batches, `n` is the number of points and `f` is the number of features.
    `f` is supposed to be 4.

    :param points:         is the point cloud whose points are scaled.
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

    :param points:    is the point cloud that contains the points that will be scaled.
    :param labels:    are the labels belonging to the aforementioned point cloud.
    :param sigma:     is the the standard deviation of the truncated normal distribution.
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

    :param points:    is the point cloud that contains the points that will be scaled.
    :param labels:    are the labels belonging to the aforementioned point cloud.
    :param sigma:     is the the standard deviation of the truncated normal distribution.
    :param max_scale: is the upper limit of the truncated normal distribution. The lower limit is the inverse.
    """

    _check_labels_and_points(points, labels)

    transformations.scale_local(points, labels, sigma, max_scale)


def flip_random(points: Tensor, labels: Tensor, prob: int) -> None:
    assert prob >= 0 and prob <= 100, f"{prob}% is not a valid probability"
    _check_labels_and_points(points, labels)

    transformations.flip_random(points, labels, prob)


def random_noise(points: Tensor, sigma: float,
                 ranges: list[float] | transformations.distribution_ranges,
                 noise_type: transformations.noise) -> None:
    _check_points(points)

    if type(ranges) is list:
        x_min, x_max = ranges[0], ranges[1]
        y_min, y_max = ranges[2], ranges[3]
        z_min, z_max = ranges[4], ranges[5]
        uniform_min, uniform_max = ranges[6], ranges[7]

        distribution_ranges = transformations.distribution_ranges(
            transformations.distribution_range(x_min, x_max),
            transformations.distribution_range(y_min, y_max),
            transformations.distribution_range(z_min, z_max),
            transformations.distribution_range(uniform_min, uniform_max))
    else:
        distribution_ranges = ranges

    transformations.random_noise(points, sigma, distribution_ranges,
                                 noise_type)


def thin_out(points: Tensor, sigma: float) -> None:
    """
     Randomly genereates a percentage from a norma distribution, which determines
     how many items should be 'thinned out'. From that percentage random indeces
     are uniformly drawn (in a random order, where each index is unique).

     Finally a new tensor is created containing the items present at those
     indeces.

    :param points: is the point cloud.
    :param sigma:  is the standard diviation of the distribution that genereates the percentage.
    """

    _check_points(points)

    batch_points: Tensor = transformations.thin_out(points, sigma)
    points = batch_points


def rotate_deg(points: Tensor, angle: float) -> None:
    """
    Rotates a batch of points anlong the 'z' axis (yaw).

    :param points: is the point cloud that the rotation is applied to.
    :param angle:  is the angle (in degrees) by which the points are to be rotated.
    """

    _check_points(points)

    transformations.rotate_deg(points, angle)


def rotate_rad(points: Tensor, angle: float) -> None:
    """
    Rotates a batch of points anlong the 'z' axis (yaw).

    :param points: is the point cloud that the rotation is applied to.
    :param angle:  is the angle (in radians) by which the points are to be rotated.
    """

    _check_points(points)

    transformations.rotate_rad(points, angle)


def rotate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    """
    Rotates points and labels.
    The number of degrees that they are rotated by is determined by a randomly genereated value from a normal distribution.

    :param points: is the point cloud that the rotation is applied to.
    :param labels: are the labels belonging to the point cloud that the rotation is applied to.
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

    :param points:     is the point_cloud.
    :param labels:     are the bounding boxes of objects.
    :param names:      are the names/labels of these boxes.
    :param min_points: is the point threshold.
    """

    _check_labels_and_points(points, labels)

    batch_labels, batch_names = transformations.delete_labels_by_min_points(
        points, labels, names, min_points)

    labels = batch_labels
    names = batch_names


def random_point_noise(points: Tensor, sigma: float):
    """
    Moves each point in the point cloud randomly.
    How much each coordinate is changed is decided by values drawn from a normal distribution.

    :param points: is the point cloud from which each point is moved.
    :param sigma:  is the standard diviation of the normal distribution.
    """

    _check_points(points)

    transformations.random_point_noise(points, sigma)


def fog(point_cloud: Tensor, prob: float, metric: transformations.fog_metric,
        sigma: float, mean: int) -> None:
    result = transformations.fog(point_cloud, prob, metric, sigma, mean)

    if result:
        point_cloud = result.value()

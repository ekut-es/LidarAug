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
    _check_points(points)

    transformations.translate(points, translation)


def translate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    _check_labels_and_points(points, labels)

    transformations.translate_random(points, labels, sigma)


def scale_random(points: Tensor, labels: Tensor, sigma: float,
                 max_scale: float) -> None:
    _check_labels_and_points(points, labels)

    transformations.scale_random(points, labels, sigma, max_scale)


def scale_local(points: Tensor, labels: Tensor, sigma: float,
                max_scale: float) -> None:
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
    _check_points(points)

    transformations.thin_out(points, sigma)


def rotate_random(points: Tensor, labels: Tensor, sigma: float) -> None:
    _check_labels_and_points(points, labels)

    transformations.rotate_random(points, labels, sigma)


def delete_labels_by_min_points(points: Tensor, labels: Tensor, names: Tensor,
                                min_points: int) -> None:
    _check_labels_and_points(points, labels)

    batch_labels, batch_names = transformations.delete_labels_by_min_points(
        points, labels, names, min_points)

    labels = batch_labels
    names = batch_names

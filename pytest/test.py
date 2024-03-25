from contextlib import nullcontext as does_not_raise
import pytest

import torch
from LidarAug import augmentations as aug
from LidarAug.transformations import NoiseType, DistributionRange, DistributionRanges, IntensityRange

import re

POINT_CLOUD_FEATRUES = 4
LABEL_FEATRUES = 7
WRONG_NUMBER_OF_FEATURES_PC = re.escape(
    "point is supposed to have 4 components (x, y, z, intensity)!")

WRONG_NUMBER_OF_FEATURES_LABEL = re.escape(
    "label is supposed to have 7 components (x, y, z, width, height, length, theta)!"
)
WRONG_SHAPE_PC = re.escape(
    "Tensor is not of shape (B, N, F), where B is the batchsize, N is the number of points and F is the number of features!"
)

WRONG_SHAPE_LABELS = re.escape(
    "Tensor is not of shape (B, N, F), where B is the batchsize, N is the number of labels and F is the number of features!"
)
INCOMPATIBLE_BATCH_SIZES = re.escape(
    "Batch sizes for points and labels are not equal!")

WRONG_FRAME_DIMENSIONS = re.escape(
    "`frame` is supposed to be a 6-vector (x, y, z, roll, yaw, pitch)")

test_points: torch.Tensor = torch.randn([3, 100, POINT_CLOUD_FEATRUES])


@pytest.mark.shapetest
@pytest.mark.parametrize(
    "tensor,expectation",
    [(torch.randn([1, 2, POINT_CLOUD_FEATRUES - 1]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_PC)),
     (torch.randn([30, 19, POINT_CLOUD_FEATRUES]), does_not_raise()),
     (torch.randn([3, 2, POINT_CLOUD_FEATRUES + 3]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_PC)),
     (torch.randn([30, POINT_CLOUD_FEATRUES
                   ]), pytest.raises(AssertionError, match=WRONG_SHAPE_PC)),
     (torch.randn([
         POINT_CLOUD_FEATRUES, POINT_CLOUD_FEATRUES, POINT_CLOUD_FEATRUES - 1
     ]), pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_PC)),
     (torch.randn([20]), pytest.raises(AssertionError, match=WRONG_SHAPE_PC))])
def test_check_points(tensor, expectation):
    with expectation:
        aug._check_points(tensor)


@pytest.mark.shapetest
@pytest.mark.parametrize(
    "tensor,expectation",
    [(torch.randn([1, 2, LABEL_FEATRUES - 1]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([1, 2, POINT_CLOUD_FEATRUES]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([30, 19, LABEL_FEATRUES]), does_not_raise()),
     (torch.randn([3, 2, LABEL_FEATRUES + 3]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([30, LABEL_FEATRUES]),
      pytest.raises(AssertionError, match=WRONG_SHAPE_LABELS)),
     (torch.randn([LABEL_FEATRUES, LABEL_FEATRUES, LABEL_FEATRUES - 1]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([20]), pytest.raises(AssertionError,
                                       match=WRONG_SHAPE_LABELS))])
def test_check_labels(tensor, expectation):
    with expectation:
        aug._check_labels(tensor)


@pytest.mark.shapetest
@pytest.mark.parametrize("points,labels,expectation", [
    (torch.randn([1, 2, POINT_CLOUD_FEATRUES
                  ]), torch.randn([1, 2, LABEL_FEATRUES]), does_not_raise()),
    (torch.randn([1, 1, POINT_CLOUD_FEATRUES
                  ]), torch.randn([1, 2, LABEL_FEATRUES]), does_not_raise()),
    (torch.randn([2, 2, POINT_CLOUD_FEATRUES
                  ]), torch.randn([1, 2, LABEL_FEATRUES]),
     pytest.raises(AssertionError, match=INCOMPATIBLE_BATCH_SIZES)),
])
def test_check_points_and_labels(points, labels, expectation):
    with expectation:
        aug._check_labels_and_points(points, labels)


@pytest.mark.shapetest
@pytest.mark.parametrize("frame,expectation", [
    (torch.randn([1, 2, POINT_CLOUD_FEATRUES]),
     pytest.raises(AssertionError, match=WRONG_FRAME_DIMENSIONS)),
    (torch.randn([1, 2, LABEL_FEATRUES]),
     pytest.raises(AssertionError, match=WRONG_FRAME_DIMENSIONS)),
    (torch.randn([6]), does_not_raise()),
    (torch.tensor([1, 2, 3, 4, 5, 6]), does_not_raise()),
])
def test_check_frame_coordinate_dimensions(frame, expectation):
    with expectation:
        aug._check_frame_coordinate_dimensions(frame)


@pytest.mark.transtest
def test_random_noise():
    points = torch.empty([1, 0, 4])
    ranges_list = [1.0, 10.0, 3.0, 5.0, 4.0, 7.0, 0.0, 10.0]
    aug.random_noise(points, 2, ranges_list, NoiseType.UNIFORM,
                     IntensityRange.MAX_INTENSITY_255)

    assert points.shape[1] > 0, "No points have been added!"
    for point in points[0]:
        assert point[0] <= 10 and point[0] >= 1, "x range not as parametrized"
        assert point[1] <= 5 and point[1] >= 3, "y range not as parametrized"
        assert point[2] <= 7 and point[2] >= 4, "z range not as parametrized"
        assert point[3] <= 255 and point[
            3] >= 0, "intensity range not as parametrized"


@pytest.mark.transtest
def test_thin_out():
    points = test_points.clone()
    aug.thin_out(points, 10)
    assert points.shape[1] != test_points.shape[1]

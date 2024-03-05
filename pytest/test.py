from contextlib import nullcontext as does_not_raise
import pytest

import torch
from LidarAug import augmentations as aug

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


@pytest.mark.xfail(reason="Not implemented")
def test_random_noise():
    assert False


@pytest.mark.xfail(reason="Not implemented")
def test_thin_out():
    assert False

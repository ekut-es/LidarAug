from contextlib import nullcontext as does_not_raise
import os
import pickle
import pytest

import torch
from lidar_aug import augmentations as aug
from lidar_aug import weather_simulations
from lidar_aug import evaluation
from lidar_aug.transformations import NoiseType
from lidar_aug.point_cloud import IntensityRange

import re

POINT_CLOUD_FEATURES = 4
LABEL_FEATURES = 7
WRONG_NUMBER_OF_FEATURES_PC = re.escape(
    "point is supposed to have 4 components (x, y, z, intensity)!")

WRONG_NUMBER_OF_FEATURES_LABEL = re.escape(
    "label is supposed to have 7 components (x, y, z, width, height, length, theta)!"
)
WRONG_SHAPE_PC = re.escape(
    "Tensor is not of shape (B, N, F), where B is the batch-size, N is the number of points and F is the number of features!"
)

WRONG_SHAPE_LABELS = re.escape(
    "Tensor is not of shape (B, N, F), where B is the batch-size, N is the number of labels and F is the number of features!"
)
INCOMPATIBLE_BATCH_SIZES = re.escape(
    "Batch sizes for points and labels are not equal!")

WRONG_FRAME_DIMENSIONS = re.escape(
    "`frame` is supposed to be a 6-vector (x, y, z, roll, yaw, pitch)")

test_points: torch.Tensor = torch.randn([3, 100, POINT_CLOUD_FEATURES])


@pytest.mark.shapetest
@pytest.mark.parametrize(
    "tensor,expectation",
    [(torch.randn([1, 2, POINT_CLOUD_FEATURES - 1]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_PC)),
     (torch.randn([30, 19, POINT_CLOUD_FEATURES]), does_not_raise()),
     (torch.randn([3, 2, POINT_CLOUD_FEATURES + 3]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_PC)),
     (torch.randn([30, POINT_CLOUD_FEATURES
                   ]), pytest.raises(AssertionError, match=WRONG_SHAPE_PC)),
     (torch.randn([
         POINT_CLOUD_FEATURES, POINT_CLOUD_FEATURES, POINT_CLOUD_FEATURES - 1
     ]), pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_PC)),
     (torch.randn([20]), pytest.raises(AssertionError, match=WRONG_SHAPE_PC))])
def test_check_points(tensor, expectation):
    with expectation:
        aug._check_points(tensor)


@pytest.mark.shapetest
@pytest.mark.parametrize(
    "tensor,expectation",
    [(torch.randn([1, 2, LABEL_FEATURES - 1]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([1, 2, POINT_CLOUD_FEATURES]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([30, 19, LABEL_FEATURES]), does_not_raise()),
     (torch.randn([3, 2, LABEL_FEATURES + 3]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([30, LABEL_FEATURES]),
      pytest.raises(AssertionError, match=WRONG_SHAPE_LABELS)),
     (torch.randn([LABEL_FEATURES, LABEL_FEATURES, LABEL_FEATURES - 1]),
      pytest.raises(AssertionError, match=WRONG_NUMBER_OF_FEATURES_LABEL)),
     (torch.randn([20]), pytest.raises(AssertionError,
                                       match=WRONG_SHAPE_LABELS))])
def test_check_labels(tensor, expectation):
    with expectation:
        aug._check_labels(tensor)


@pytest.mark.shapetest
@pytest.mark.parametrize("points,labels,expectation", [
    (torch.randn([1, 2, POINT_CLOUD_FEATURES
                  ]), torch.randn([1, 2, LABEL_FEATURES]), does_not_raise()),
    (torch.randn([1, 1, POINT_CLOUD_FEATURES
                  ]), torch.randn([1, 2, LABEL_FEATURES]), does_not_raise()),
    (torch.randn([2, 2, POINT_CLOUD_FEATURES
                  ]), torch.randn([1, 2, LABEL_FEATURES]),
     pytest.raises(AssertionError, match=INCOMPATIBLE_BATCH_SIZES)),
])
def test_check_points_and_labels(points, labels, expectation):
    with expectation:
        aug._check_labels_and_points(points, labels)


@pytest.mark.shapetest
@pytest.mark.parametrize("frame,expectation", [
    (torch.randn([1, 2, POINT_CLOUD_FEATURES]),
     pytest.raises(AssertionError, match=WRONG_FRAME_DIMENSIONS)),
    (torch.randn([1, 2, LABEL_FEATURES]),
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
    aug.random_noise(points, 100, ranges_list, NoiseType.UNIFORM,
                     IntensityRange.MAX_INTENSITY_255)

    assert points.shape[1] > 0, "No points have been added!"
    for point in points[0]:
        assert 10 >= point[0] >= 1, "x range not as parametrized"
        assert 5 >= point[1] >= 3, "y range not as parametrized"
        assert 7 >= point[2] >= 4, "z range not as parametrized"
        assert 255 >= point[3] >= 0, "intensity range not as parametrized"


@pytest.mark.transtest
def test_thin_out():
    points = test_points.clone()
    aug.thin_out(points, 10)
    assert points.shape[1] != test_points.shape[1]


@pytest.mark.transtest
def test_delete_labels_by_min_points():
    points = torch.tensor([[[-8.2224, -4.3151, -6.5488, -3.9899],
                            [6.3092, -3.7737, 7.2516, -5.8651],
                            [1.0, 1.0, 1.0, 10.0]],
                           [[10.4966, 10.1144, 10.2182, -8.4158],
                            [7.0241, 7.6908, -2.1535, 1.3416],
                            [10.0, 10.0, 10.0, 10.0]]])

    labels = torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 0.0]],
                           [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                            [10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0]]])
    names = torch.tensor([[[0x00], [0x01]], [[0x10], [0x11]]])

    min_points = 1
    aug.delete_labels_by_min_points(points, labels, names, min_points)

    expected_points = torch.tensor([[[-8.2224, -4.3151, -6.5488, -3.9899],
                                     [6.3092, -3.7737, 7.2516, -5.8651],
                                     [1.0, 1.0, 1.0, 10.0]],
                                    [[10.4966, 10.1144, 10.2182, -8.4158],
                                     [7.0241, 7.6908, -2.1535, 1.3416],
                                     [10.0, 10.0, 10.0, 10.0]]])

    expected_labels = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
         [10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0, 1.0]])

    expected_names = torch.tensor([[0x00, 0], [0x11, 1]])

    assert points.equal(
        expected_points), "Points should not have been modified!"

    assert labels.equal(expected_labels)

    assert names.equal(expected_names)


def check_precision(val1: float, val2: float, precision: int) -> None:
    """
    Asserts that two floating point values are equal up a certain amount of digits after the comma.

    :param val1:       First comparison value.
    :param val2:       Second comparison value.
    :param precision:  The number of significant digits after the comma.
    """

    multiplier = 10**precision

    assert int(val1 * multiplier) == int(val2 * multiplier)


@pytest.mark.evaltest
def test_result_dict_factory():
    expected = {
        3: {
            'tp': [],
            'fp': [],
            'gt': [0],
            'score': []
        },
        5: {
            'tp': [],
            'fp': [],
            'gt': [0],
            'score': []
        },
        7: {
            'tp': [],
            'fp': [],
            'gt': [0],
            'score': []
        }
    }

    r = evaluation.make_result_dict(expected)

    assert dict(r) == expected


@pytest.mark.evaltest
def test_evaluate():
    data_path = "./pytest/data/pkl/"

    files = os.listdir(data_path)

    for file in files:
        path = data_path + file
        with open(path, 'rb') as f:
            dump_dict: dict = pickle.load(f)

            results = evaluation.make_result_dict(
                dump_dict["result_stat_tp_fp"])
            aps = evaluation.evaluate(results, False)

            expected_aps = dump_dict["ap_dict"]
            ap_keys = list(expected_aps.keys())

            ap_30_key = ap_keys[0]
            ap_50_key = ap_keys[1]
            ap_70_key = ap_keys[2]

            ap_30 = expected_aps[ap_30_key]
            ap_50 = expected_aps[ap_50_key]
            ap_70 = expected_aps[ap_70_key]

            check_precision(aps[0], ap_30, 4)
            check_precision(aps[1], ap_50, 4)
            check_precision(aps[2], ap_70, 4)


@pytest.mark.evaltest
def test_false_and_true_positive():
    result_stat_template = {
        3: {
            'tp': [],
            'fp': [],
            'gt': [0],
            'score': []
        },
        5: {
            'tp': [],
            'fp': [],
            'gt': [0],
            'score': []
        },
        7: {
            'tp': [],
            'fp': [],
            'gt': [0],
            'score': []
        }
    }

    thresholds = [.3, .5, .7]

    data_path = "./pytest/data/pkl/"

    files = os.listdir(data_path)

    for file in files:

        print(f"file: {file}")

        path = data_path + file
        with open(path, 'rb') as f:

            print(f"\n----- file: {file} -----\n")

            dump_dict: dict = pickle.load(f)

            result_stat = evaluation.make_result_dict(result_stat_template)

            for threshold in thresholds:

                print(f"\n------ threshold: {threshold} -----\n")

                for i, gt_anno in enumerate(dump_dict["gt_anno"]):

                    boxes_lidar = torch.from_numpy(
                        dump_dict["det_anno"][i]['boxes_lidar'])
                    score = torch.from_numpy(dump_dict["det_anno"][i]['score'])
                    gt = torch.from_numpy(gt_anno)

                    if len(gt) == 0 or len(boxes_lidar) == 0:
                        continue

                    evaluation.calculate_false_and_true_positive_2d(
                        boxes_lidar, score, gt, threshold, result_stat)

                expected = dump_dict["result_stat_tp_fp"][int(threshold * 10)]

                result = result_stat[int(threshold * 10)]

                assert len(result['fp']) == len(expected['fp'])
                assert len(result['tp']) == len(expected['tp'])
                assert len(result['score']) == len(expected['score'])
                assert result['gt'][0] == expected['gt'][0]
                assert result['fp'] == expected['fp']
                assert result['tp'] == expected['tp']

                for result_score, expected_score in zip(
                        result['score'], expected['score']):
                    check_precision(result_score, expected_score, 2)


@pytest.mark.weathertest
def test_fog():
    points = torch.randn([100, 4])
    metric = weather_simulations.FogParameter.DIST
    viewing_dist = 100

    result = weather_simulations.fog(points, metric, viewing_dist,
                                     IntensityRange.MAX_INTENSITY_1)

    assert not result.equal(points)


@pytest.mark.weathertest
def test_fog_100():
    points = torch.randn([1, 100, 4])
    prob = 100.0
    metric = weather_simulations.FogParameter.DIST
    sigma = 1.0
    mean = 5

    result = weather_simulations.fog(points, prob, metric, sigma, mean)

    assert result is not None

    for i, tensor in enumerate(result):
        assert not tensor.equal(points[i])


def test_fog_0():
    points = torch.randn([1, 100, 4])
    prob = 0
    metric = weather_simulations.FogParameter.DIST
    sigma = 1.0
    mean = 5

    result = weather_simulations.fog(points, prob, metric, sigma, mean)

    assert result is None

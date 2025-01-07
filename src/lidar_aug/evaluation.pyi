import pybind11_stubgen.typing_ext
import torch
import typing

__all__ = [
    'ResultDict', 'calculate_false_and_true_positive_2d',
    'calculate_false_and_true_positive_3d', 'evaluate', 'make_result_dict'
]


class ResultDict:
    """
    Wrapping type around a C++ `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>`.

    Converts into a Python `dict[int, dict[str, list[float]]]`.
    """

    def __bool__(self) -> bool:
        """
        Check whether the map is nonempty
        """

    @typing.overload
    def __contains__(self, arg0: int) -> bool:
        ...

    @typing.overload
    def __contains__(self, arg0: typing.Any) -> bool:
        ...

    def __delitem__(self, arg0: int) -> None:
        ...

    def __getitem__(self, arg0: int) -> dict[str, list[float]]:
        ...

    def __init__(self) -> None:
        ...

    def __iter__(self) -> typing.Iterator:
        ...

    def __len__(self) -> int:
        ...

    def __repr__(self) -> str:
        """
        Return the canonical string representation of this map.
        """

    def __setitem__(self, arg0: int, arg1: dict[str, list[float]]) -> None:
        ...

    def items(self) -> typing.ItemsView[int, dict[str, list[float]]]:
        ...

    def keys(self) -> typing.KeysView[int]:
        ...

    def values(self) -> typing.ValuesView[dict[str, list[float]]]:
        ...


def calculate_false_and_true_positive_2d(detection_boxes: torch.Tensor,
                                         detection_score: torch.Tensor,
                                         ground_truth_box: torch.Tensor,
                                         iou_threshold: float,
                                         results: ResultDict) -> None:
    """
    Calculates the false and true positive rate of a set of predictions against a set of ground truth binding boxes by calculating the 'intersection over union' (IOU) for 2D boxes.
    The results are saved in a `result_dict` structure.

    :param detection_boxes: The 2D object detection box.
    :param detection_score: The detection scores used to index the detection boxes.
    :param ground_truth_box: The 2D ground truth box containing the actual object.
    :param iou_threshold: The threshold that determines wether the prediction is accurate or not.
    :param results: A `ResultDict` that is filled with the results of the calculations.
    """
    ...


def calculate_false_and_true_positive_3d(detection_boxes: torch.Tensor,
                                         detection_score: torch.Tensor,
                                         ground_truth_box: torch.Tensor,
                                         iou_threshold: float,
                                         results: ResultDict) -> None:
    """
    Calculates the false and true positive rate of a set of predictions against a set of ground truth binding boxes by calculating the 'intersection over union' (IOU) for 3D boxes.
    The results are saved in a `result_dict` structure.

    :param detection_boxes: The 3D object detection box.
    :param detection_score: The detection scores used to index the detection boxes.
    :param ground_truth_box: The 3D ground truth box containing the actual object.
    :param iou_threshold: The threshold that determines wether the prediction is accurate or not.
    :param results: A `ResultDict` that is filled with the results of the calculations.
    """
    ...


def evaluate(
    results: ResultDict, global_sort_detections: bool
) -> typing.Annotated[list[float],
                      pybind11_stubgen.typing_ext.FixedSize(3)]:
    """
    Calculages the average precision of a set of results with the IOU thresholds of 0.3, 0.5 & 0.7.

    :param results: The results for which the average precision is calculated
    :param global_sort_detections: Enables/Disables the sorting of true and false positive values
    :return: A list with the average precision values for the IOU thresholds of 0.3, 0.5 & 0.7.
    """
    ...


def make_result_dict(input: dict) -> ResultDict:
    """
    Create a `result_dict` aka `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>` from a `dict[int, dict[str, list[float]]]`.

    :param input: A Python `dict[int, dict[str, list[float]]]`.
    :return: A `ResultDict` (C++ `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>`).
    """
    ...

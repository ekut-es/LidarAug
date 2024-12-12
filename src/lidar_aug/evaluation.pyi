from torch import Tensor


def evaluate(results: ResultDict, global_sort_detections: bool) -> list[float]:
    """
    Calculages the average precision of a set of results with the IOU thresholds of 0.3, 0.5 & 0.7.

    :param results: The results for which the average precision is calculated
    :param global_sort_detections: Enables/Disables the sorting of true and false positive values
    :return: A list with the average precision values for the IOU thresholds of 0.3, 0.5 & 0.7.
    """
    ...


class ResultDict:
    """
    Wrapping type around a
    C++ `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>`.

    Converts into a Python `dict[int, dict[str, list[float]]]`.
    """

    def __init__(self):
        ...

    def __iter__(self):
        ...

    def __getitem__(self, key):
        ...


def make_result_dict(input: dict[int, dict[str, list[float]]]) -> ResultDict:
    """
    Create a `result_dict` aka `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>` from a `dict[int, dict[str, list[float]]]`.

    :param input: A Python `dict[int, dict[str, list[float]]]`.
    :return: A `ResultDict` (C++ `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>`).
    """
    ...


def calculate_false_and_true_positive_2d(detection_boxes: Tensor,
                                         detection_score: Tensor,
                                         ground_truth_box: Tensor,
                                         iou_threshold: float,
                                         results: ResultDict):
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


def calculate_false_and_true_positive_3d(detection_boxes: Tensor,
                                         detection_score: Tensor,
                                         ground_truth_box: Tensor,
                                         iou_threshold: float,
                                         results: ResultDict):
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

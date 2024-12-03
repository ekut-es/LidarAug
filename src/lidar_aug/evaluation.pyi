from torch import Tensor


def evaluate(results: result_dict,
             global_sort_detections: bool) -> list[float]:
    ...


class result_dict:
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


def make_result_dict(input: dict[int, dict[str, list[float]]]) -> result_dict:
    """
    Create a `result_dict` aka `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>` from a `dict[int, dict[str, list[float]]]`.

    :param input: A Python `dict[int, dict[str, list[float]]]`.
    :return: A `result_dict` (C++ `std::map<std::uint8_t, std::map<std::string, std::vector<float>>>`).
    """
    ...


def calculate_false_and_true_positive_2d(detection_boxes: Tensor,
                                         detection_score: Tensor,
                                         ground_truth_box: Tensor,
                                         iou_threshold: float,
                                         results: result_dict):
    ...


def calculate_false_and_true_positive_3d(detection_boxes: Tensor,
                                         detection_score: Tensor,
                                         ground_truth_box: Tensor,
                                         iou_threshold: float,
                                         results: result_dict):
    ...

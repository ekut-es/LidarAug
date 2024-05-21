from torch import Tensor


def evaluate(results: dict[float, dict[str, list[float]]],
             global_sort_detections: bool) -> list[float]:
    ...


class result_dict:

    def __init__(self):
        ...

    def __iter__(self):
        ...


def make_result_dict(input: dict[int, dict[str, list[float]]]) -> result_dict:
    ...


def calculate_false_and_true_positive(detection_boxes: Tensor,
                                      detection_score: Tensor,
                                      ground_truth_box: Tensor,
                                      iou_threshold: float,
                                      results: result_dict):
    ...

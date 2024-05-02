from torch import Tensor


def evaluate(results: dict[float, dict[str, list]], dir: str,
             global_sort_detections: bool):
    ...


def calculate_false_and_true_positive(detection_boxes: Tensor,
                                      detection_score: Tensor,
                                      ground_truth_box: Tensor,
                                      iou_threshold: float,
                                      results: dict[float, dict[str, list]]):
    ...

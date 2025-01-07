#include "evaluation.hpp"
#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using arg = pybind11::arg;

PYBIND11_MAKE_OPAQUE(result_dict);

result_dict make_result_dict(const pybind11::dict &dict) {
  result_dict result;

  for (auto &item : dict) {
    auto iou_threshold = item.first.cast<std::uint8_t>();

    auto inner_dict = item.second.cast<pybind11::dict>();

    std::map<std::string, std::vector<float>> results;

    for (auto &inner_item : inner_dict) {
      auto metric = inner_item.first.cast<std::string>();
      auto vector = inner_item.second.cast<std::vector<float>>();
      results[metric] = vector;
    }

    result[iou_threshold] = results;
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluate", &evaluate_results, arg("results"),
        arg("global_sort_detections"),
        "Calculages the average precision of a set of results with the IOU "
        "thresholds of 0.3, 0.5 & 0.7.\n"
        "\n"
        ":param results: The results for which the average precision is "
        "calculated\n"
        ":param global_sort_detections: Enables/Disables the sorting of true "
        "and false positive values\n"
        ":return: A list with the average precision values for the IOU "
        "thresholds of 0.3, 0.5 & 0.7.\n");

  m.def("calculate_false_and_true_positive_2d",
        calculate_false_and_true_positive<evaluation_utils::point2d_t>,
        arg("detection_boxes"), arg("detection_score"), arg("ground_truth_box"),
        arg("iou_threshold"), arg("results"),
        "Calculates the false and true positive rate of a set of predictions "
        "against a set of ground truth binding boxes by calculating the "
        "'intersection over union' (IOU) for 2D boxes.\n"
        "The results are saved in a `result_dict` structure.\n"
        "\n"
        ":param detection_boxes: The 2D object detection box.\n"
        ":param detection_score: The detection scores used to index the "
        "detection boxes.\n"
        ":param ground_truth_box: The 2D ground truth box containing the "
        "actual object.\n"
        ":param iou_threshold: The threshold that determines wether the "
        "prediction is accurate or not.\n"
        ":param results: A `ResultDict` that is filled with the results of the "
        "calculations.");

  m.def("calculate_false_and_true_positive_3d",
        calculate_false_and_true_positive<evaluation_utils::point3d_t>,
        arg("detection_boxes"), arg("detection_score"), arg("ground_truth_box"),
        arg("iou_threshold"), arg("results"),
        "Calculates the false and true positive rate of a set of predictions "
        "against a set of ground truth binding boxes by calculating the "
        "'intersection over union' (IOU) for 3D boxes.\n"
        "The results are saved in a `result_dict` structure.\n"
        "\n"
        ":param detection_boxes: The 3D object detection box.\n"
        ":param detection_score: The detection scores used to index the "
        "detection boxes.\n"
        ":param ground_truth_box: The 3D ground truth box containing the "
        "actual object.\n"
        ":param iou_threshold: The threshold that determines wether the "
        "prediction is accurate or not.\n"
        ":param results: A `ResultDict` that is filled with the results of the "
        "calculations.");

  m.def("make_result_dict", &make_result_dict, arg("input"),
        "Create a `result_dict` aka `std::map<std::uint8_t, "
        "std::map<std::string, std::vector<float>>>` from a `dict[int, "
        "dict[str, list[float]]]`.\n"
        "\n"
        ":param input: A Python `dict[int, dict[str, list[float]]]`.\n"
        ":return: A `ResultDict` (C++ `std::map<std::uint8_t, "
        "std::map<std::string, std::vector<float>>>`).\n");

  pybind11::bind_map<result_dict>(
      m, "ResultDict",
      "Wrapping type around a C++ `std::map<std::uint8_t, "
      "std::map<std::string, std::vector<float>>>`.\n"
      "\n"
      "Converts into a Python `dict[int, dict[str, list[float]]]`.\n");
}

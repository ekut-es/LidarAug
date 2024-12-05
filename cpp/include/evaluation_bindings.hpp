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
        arg("global_sort_detections"));

  m.def("calculate_false_and_true_positive_2d",
        calculate_false_and_true_positive<evaluation_utils::point2d_t>,
        arg("detection_boxes"), arg("detection_score"), arg("ground_truth_box"),
        arg("iou_threshold"), arg("results"));

  m.def("calculate_false_and_true_positive_3d",
        calculate_false_and_true_positive<evaluation_utils::point3d_t>,
        arg("detection_boxes"), arg("detection_score"), arg("ground_truth_box"),
        arg("iou_threshold"), arg("results"));

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

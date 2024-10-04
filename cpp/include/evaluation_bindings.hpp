#include "evaluation.hpp"
#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

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
  m.def("evaluate", &evaluate_results);
  m.def("calculate_false_and_true_positive_2d",
        calculate_false_and_true_positive<evaluation_utils::point2d_t>);
  m.def("calculate_false_and_true_positive_3d",
        calculate_false_and_true_positive<evaluation_utils::point3d_t>);
  m.def("make_result_dict", &make_result_dict);
  pybind11::bind_map<result_dict>(m, "result_dict");
}

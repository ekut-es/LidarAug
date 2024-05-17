#include "evaluation.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(result_dict);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluate", &evaluate_results);
  m.def("calculate_false_and_true_positive",
        &calculate_false_and_true_positive);
  pybind11::bind_map<result_dict>(m, "result_dict");
}

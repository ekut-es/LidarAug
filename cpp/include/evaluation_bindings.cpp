#include "evaluation.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluate", &evaluate_results<float>);
  m.def("calculate_false_and_true_positive",
        &calculate_false_and_true_positive);
}

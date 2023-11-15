
#include "../include/transformations.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("translate", &translate,
        "translation function for point clouds in C++");
  m.def(
      "translate_random", &translate_random,
      "random translation function for point clouds and bounding boxes in C++");
}

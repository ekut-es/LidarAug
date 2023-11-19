
#include "../include/transformations.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("translate", &translate,
        "translation function for point clouds in C++");
  m.def(
      "translate_random", &translate_random,
      "random translation function for point clouds and bounding boxes in C++");
  m.def("scale_points", &scale_points, "function to scale point clouds in C++");
  m.def("scale_labels", &scale_labels, "function to scale labels in C++");
  m.def("scale_random", &scale_random,
        "function to randomly scale point clouds and bounding boxes in C++");
  m.def("flip_random", &flip_random,
        "function to randomly flip point clouds and bounding boxes in C++");
  m.def("rotate_random", &rotate_random,
        "function to randomly rotate point clouds and bounding boxes around "
        "the 'y' axis in C++");
}

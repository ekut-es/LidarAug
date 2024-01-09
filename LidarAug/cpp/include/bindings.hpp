
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
  m.def("scale_random", &scale_random,
        "function to randomly scale point clouds and bounding boxes in C++");
  m.def("scale_local", &scale_local, "TODO");
  m.def("flip_random", &flip_random,
        "function to randomly flip point clouds and bounding boxes in C++");
  m.def("rotate_random", &rotate_random,
        "function to randomly rotate point clouds and bounding boxes around "
        "the 'y' axis in C++");
  m.def("thin_out", &thin_out,
        "function to remove a random percentage of points from a point cloud "
        "in C++");
  m.def("random_noise", &random_noise,
        "function to introduce random noise to a point clouds in C++");
  m.def("delete_labels_by_min_points", &delete_labels_by_min_points,
        "function to delete bounding boxes and their names that don't meet a "
        "minimum point threshold in C++",
        py::return_value_policy::reference_internal);
  pybind11::enum_<noise>(m, "noise")
      .value("UNIFORM", UNIFORM)
      .value("SALT_PEPPER", SALT_PEPPER)
      .value("MIN", MIN)
      .value("MAX", MAX)
      .export_values();

  // NOTE(tom): Unfortunately it is necessary to export this with defined types,
  //            as PyBind does not appear to support generics/templates.
  pybind11::class_<range<float>>(m, "distribution_range")
      .def(pybind11::init<>());
  pybind11::class_<distribution_ranges<float>>(m, "distribution_ranges")
      .def(pybind11::init<>());
}

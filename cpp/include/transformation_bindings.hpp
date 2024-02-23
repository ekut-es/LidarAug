
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
  m.def("rotate_deg", &rotate_deg,
        "function to apply a yaw rotation to points of a point cloud in C++");
  m.def("rotate_rad", &rotate_rad,
        "function to apply a yaw rotation to points of a point cloud in C++");
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
  m.def("random_point_noise", &random_point_noise,
        "Function to add random noise to point coordinates");
  m.def("transform_along_ray", &transform_along_ray,
        "Function to add move points along a ray");
  m.def("intensity_noise", &intensity_noise,
        "Introduce random noise in the intensity values");
  m.def("intensity_shift", &intensity_shift,
        "Shift intensities by a constant, random amount");
  pybind11::enum_<noise_type>(m, "NoiseType")
      .value("UNIFORM", UNIFORM)
      .value("SALT_PEPPER", SALT_PEPPER)
      .value("MIN", MIN)
      .value("MAX", MAX)
      .export_values();
  pybind11::enum_<intensity_range>(m, "IntensityRange")
      .value("MAX_INTENSITY_1", MAX_INTENSITY_1)
      .value("MAX_INTENSITY_255", MAX_INTENSITY_255)
      .export_values();

  // NOTE(tom): Unfortunately it is necessary to export this with defined types,
  //            as PyBind does not appear to support generics/templates.
  pybind11::class_<range<float>>(m, "DistributionRange")
      .def(pybind11::init<>());
  pybind11::class_<distribution_ranges<float>>(m, "DistributionRanges")
      .def(pybind11::init<>());
}

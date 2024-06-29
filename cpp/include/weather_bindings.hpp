
#include "../include/weather.hpp"
#include <ATen/core/TensorBody.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fog",
        pybind11::overload_cast<const torch::Tensor &, float, fog_parameter,
                                float, int>(&fog),
        "fog weather simulation");
  m.def(
      "fog",
      pybind11::overload_cast<torch::Tensor, fog_parameter, float, float>(&fog),
      pybind11::arg("max_intensity") = 1, "fog weather simulation");

  m.def("snow", &snow, pybind11::arg("max_intensity") = 1,
        "snow weather simulation");
  m.def("rain", &rain, "rain weather simulation");

  pybind11::enum_<fog_parameter>(m, "FogParameter")
      .value("DIST", DIST)
      .value("CHAMFER", CHAMFER)
      .export_values();
  pybind11::enum_<distribution>(m, "Distribution")
      .value("EXPONENTIAL", EXPONENTIAL)
      .value("LOG_NORMAL", LOG_NORMAL)
      .value("GM", GM)
      .export_values();
}

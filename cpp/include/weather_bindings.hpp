
#include "../include/weather.hpp"
#include <ATen/core/TensorBody.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

using arg = pybind11::arg;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fog",
        pybind11::overload_cast<const torch::Tensor &, float, fog_parameter,
                                float, int>(&fog),
        "fog weather simulation");
  m.def(
      "fog",
      pybind11::overload_cast<torch::Tensor, fog_parameter, float, float>(&fog),
      arg("point_cloud"), arg("metric"), arg("viewing_dist"),
      arg("max_intensity") = 1, "fog weather simulation");

  m.def("snow",
        pybind11::overload_cast<torch::Tensor, std::array<float, 6>, uint32_t,
                                float, int32_t, float>(&snow),
        arg("point_cloud"), arg("dims"), arg("num_drops"), arg("precipitation"),
        arg("scale"), arg("max_intensity") = 1, "snow weather simulation");
  m.def("snow",
        pybind11::overload_cast<torch::Tensor, std::string_view, uint32_t,
                                float, int32_t, float>(&snow),
        "snow weather simulation");
  m.def("rain",
        pybind11::overload_cast<torch::Tensor, std::array<float, 6>, uint32_t,
                                float, distribution>(&rain),
        "rain weather simulation");
  m.def("rain",
        pybind11::overload_cast<torch::Tensor, std::string_view, uint32_t,
                                float, float>(&rain),
        "rain weather simulation");

  pybind11::enum_<fog_parameter>(m, "FogParameter")
      .value("DIST", fog_parameter::DIST)
      .value("CHAMFER", fog_parameter::CHAMFER)
      .export_values();
  pybind11::enum_<distribution>(m, "Distribution")
      .value("EXPONENTIAL", distribution::exponential)
      .value("LOG_NORMAL", distribution::log_normal)
      .value("GM", distribution::gm)
      .export_values();
}

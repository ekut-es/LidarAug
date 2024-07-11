
#include "../include/weather.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fog", &fog, "fog weather simulation");
  pybind11::enum_<fog_parameter>(m, "FogParameter")
      .value("DIST", fog_parameter::DIST)
      .value("CHAMFER", fog_parameter::CHAMFER)
      .export_values();
}

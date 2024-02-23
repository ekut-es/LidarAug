
#include "../include/weather.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fog", &fog, "fog weather simulation");
  pybind11::enum_<fog_metric>(m, "FogMetric")
      .value("DIST", DIST)
      .value("CHAMFER", CHAMFER)
      .export_values();
}

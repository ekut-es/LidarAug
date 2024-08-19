#include "point_cloud.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_max_intensity", &point_cloud_data::max_intensity::set);
  m.def("get_max_intensity", &point_cloud_data::max_intensity::get);
}

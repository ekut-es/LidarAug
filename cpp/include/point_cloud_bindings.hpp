#include "point_cloud.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_max_intensity", &point_cloud_data::max_intensity::set);
  m.def("get_max_intensity", &point_cloud_data::max_intensity::get);

  pybind11::enum_<point_cloud_data::intensity_range>(m, "IntensityRange")
      .value("MAX_INTENSITY_1",
             point_cloud_data::intensity_range::MAX_INTENSITY_1)
      .value("MAX_INTENSITY_255",
             point_cloud_data::intensity_range::MAX_INTENSITY_255)
      .export_values();
}

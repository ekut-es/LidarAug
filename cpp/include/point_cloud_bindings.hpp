#include "point_cloud.hpp"
#include <pybind11/pybind11.h>

using arg = pybind11::arg;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::enum_<point_cloud_data::intensity_range>(
      m, "IntensityRange",
      "Defines options for maximum intensity values.\nIntensity goes from [0; "
      "MAX_INTENSITY], where MAX_INTENSITY is either 1 or 255.")
      .value("MAX_INTENSITY_1",
             point_cloud_data::intensity_range::MAX_INTENSITY_1)
      .value("MAX_INTENSITY_255",
             point_cloud_data::intensity_range::MAX_INTENSITY_255);

  m.def("set_max_intensity", &point_cloud_data::max_intensity::set, arg("val"),
        "Set the global state tracker for the maximum intensity.\n\n\
\
:param val: is the new maximum intensity (member of `IntensityRange`).");

  m.def(
      "get_max_intensity", &point_cloud_data::max_intensity::get,
      "Get the current value of the maximum intensity global state tracker.\n\n\
\
:return: an int representing the maximum intensity value.");
}

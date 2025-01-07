
#include "../include/point_cloud.hpp"
#include "../include/weather.hpp"
#include <ATen/core/TensorBody.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

using arg = pybind11::arg;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::enum_<fog_parameter>(
      m, "FogParameter", "Different parameters for the fog model/simulation.")
      .value("DIST", fog_parameter::DIST,
             "Optimization of the distance distribution between the points.")
      .value("CHAMFER", fog_parameter::CHAMFER,
             "Optimization of the chamfer distance.");

  pybind11::enum_<distribution>(
      m, "Distribution",
      "Different options to determine which statistical distribution should"
      "be used to sample the particles for some weather simulations.")
      .value("EXPONENTIAL", distribution::exponential,
             "Exponential distribution.")
      .value("LOG_NORMAL", distribution::log_normal, "Log normal distribution.")
      .value("GM", distribution::gm, "GM distribution.");

  m.def(
      "fog",
      pybind11::overload_cast<const torch::Tensor &, float, fog_parameter,
                              float, int>(&fog),
      arg("point_cloud"), arg("prob"), arg("metric"), arg("sigma"), arg("mean"),
      "Applies a fog simulation to a point cloud with a chance of `prob` %.\n"
      "The point cloud has the shape `(B, N, F)` where `B` is the number of "
      "batches, `N` is the number of points and `F` is the number of "
      "features, which is 4; `(x, y, z, i)`.\n"
      "\n"
      ":param point_cloud: is the point cloud that the simulation is applied "
      "to.\n"
      ":param prob: is the probability with which the simulation is applied.\n"
      ":param metric: is a parameter used to control the simulation.\n"
      ":param sigma: is the standard deviation used to draw the viewing "
      "distance in the fog.\n"
      ":param mean: is the mean that is used to draw the viewing distance in "
      "the fog.\n"
      ":return: A list of B tensors with P points that the simulation has been "
      "applied to or None.");

  m.def(
      "fog",
      pybind11::overload_cast<torch::Tensor, fog_parameter, float,
                              point_cloud_data::intensity_range>(&fog),
      arg("point_cloud"), arg("metric"), arg("viewing_dist"),
      arg("max_intensity") = point_cloud_data::intensity_range::MAX_INTENSITY_1,
      "Applies a fog simulation to a point cloud.\n"
      "The point cloud has the shape `(N, F)` where `N` is the number of "
      "points and `F` is the number of features, which is 4; `(x, y, z, i)`.\n"
      "\n"
      ":param point_cloud: is the point cloud that the simulation is applied "
      "to.\n"
      ":param metric: is a parameter used to control the simulation.\n"
      ":param viewing_dist:  is the viewing distance in the fog.\n"
      ":param max_intensity:  is the maximum intensity value of a point.\n"
      ":return: a new point cloud with the old points as a base but after "
      "applying the simulation.\n");

  m.def(
      "snow",
      pybind11::overload_cast<
          torch::Tensor, const cpp_utils::distribution_ranges<float> &,
          uint32_t, float, int32_t, point_cloud_data::intensity_range>(&snow),
      arg("point_cloud"), arg("dims"), arg("num_drops"), arg("precipitation"),
      arg("scale"),
      arg("max_intensity") = point_cloud_data::intensity_range::MAX_INTENSITY_1,
      "Applies a snow simulation to a point cloud.\n"
      "The point cloud has the shape `(N, F)` where `N` is the number of "
      "points and `F` is the number of features, which is 4; `(x, y, z, i)`.\n"
      "\n"
      ":param point_cloud: is the point cloud that the simulation is applied "
      "to.\n"
      ":param dims: set the upper and lower bounds of the uniform "
      "distribution used to draw new points for the noise filter.\n"
      ":param num_drops: is the number of snowflakes per m^3.\n"
      ":param precipitation: is the precipitation rate and determines the "
      "snowflake size distribution.\n"
      ":param scale: is used to scale the size of the sampled particles when "
      "generating the noise filter.\n"
      ":param max_intensity: is the maximum intensity of the points in the "
      "point cloud.\n"
      ":return: a new point cloud with the old points as a base but after "
      "applying the simulation.\n");

  m.def(
      "snow",
      pybind11::overload_cast<torch::Tensor, std::string_view, uint32_t, float,
                              int32_t, float>(&snow),
      arg("point_cloud"), arg("noise_filter_path"), arg("num_drops_sigma"),
      arg("precipitation_sigma"), arg("scale"), arg("prob"),
      "Applies a snow simulation to a point cloud with a chance of `prob` %.\n"
      "The point cloud has the shape `(N, F)` where `N` is the number of "
      "points and `F` is the number of features, which is 4; `(x, y, z, i)`.\n"
      "\n"
      ":param point_cloud: is the point cloud that the simulation is applied "
      "to.\n"
      ":param noise_filter_path: is the path to the directory containing the "
      "npz files with the noise filter data.\n"
      ":param num_drops_sigma: is the standard deviation for the number of "
      "snowflakes (used to find the correct noise filter).\n"
      ":param precipitation_sigma: is the standard deviation for the "
      "precipitation rate (used to find the correct noise filter).\n"
      ":param scale: is used to scale the size of the sampled particles when "
      "generating the noise filter.\n"
      ":param prob: is the probability that the simulation will be executed.\n"
      ":return: a new point cloud with the old points as a base but after "
      "applying the simulation.\n");

  m.def(
      "rain",
      pybind11::overload_cast<
          torch::Tensor, const cpp_utils::distribution_ranges<float> &,
          uint32_t, float, distribution, point_cloud_data::intensity_range>(
          &rain),
      arg("point_cloud"), arg("dims"), arg("num_drops"), arg("precipitation"),
      arg("d"),
      arg("max_intensity") = point_cloud_data::intensity_range::MAX_INTENSITY_1,
      "Applies a rain simulation to a point cloud.\n"
      "The point cloud has the shape `(N, F)` where `N` is the number of "
      "points and `F` is the number of features, which is 4; `(x, y, z, i)`.\n"
      "\n"
      ":param point_cloud: is the point cloud that the simulation is applied "
      "to.\n"
      ":param dims: set the upper and lower bounds of the uniform "
      "distribution used to draw new points for the noise filter.\n"
      ":param num_drops: is the number of rain drops per m^3.\n"
      ":param precipitation: is the precipitation rate and determines the "
      "raindrop size distribution.\n"
      ":param d: is the distribution function used when sampling the "
      "particles.\n"
      ":param max_intensity: is the maximum intensity of the points in the "
      "point cloud.\n"
      ":return: a new point cloud with the old points as a base but after "
      "applying the simulation.\n");

  m.def(
      "rain",
      pybind11::overload_cast<torch::Tensor, std::string_view, uint32_t, float,
                              float>(&rain),
      arg("point_cloud"), arg("noise_filter_path"), arg("num_drops_sigma"),
      arg("precipitation_sigma"), arg("prob"),
      "Applies a rain simulation to a point cloud with a chance of `prob` %.\n"
      "The point cloud has the shape `(N, F)` where `N` is the number of "
      "points and `F` is the number of features, which is 4; `(x, y, z, i)`.\n"
      "\n"
      ":param point_cloud: is the point cloud that the simulation is applied "
      "to.\n"
      ":param noise_filter_path: is the path to the directory containing the "
      "npz files with the noise filter data.\n"
      ":param num_drops_sigma: is the standard deviation for the number of "
      "drops (used to find the correct noise filter).\n"
      ":param precipitation_sigma: is the standard deviation for the "
      "precipitation rate (used to find the correct noise filter).\n"
      ":param prob: is the probability that the simulation will be executed.\n"
      ":return: a new point cloud with the old points as a base but after "
      "applying the simulation.\n");
}

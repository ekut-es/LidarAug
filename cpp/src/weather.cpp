
#include "../include/weather.hpp"
#include "../include/stats.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/pow.h>
#include <cmath>
#include <cnpy/cnpy.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <tuple>

using namespace torch::indexing;
using namespace torch_utils;

[[nodiscard]] inline std::tuple<float, float, float>
calculate_factors(const fog_parameter metric, const float viewing_dist) {
  switch (metric) {
  case DIST: {
    const float extinction_factor = 0.32f * exp(-0.022 * viewing_dist);
    const float beta = (-0.00846f * viewing_dist) + 2.29;
    const float delete_probability = -0.63f * exp(-0.02 * viewing_dist) + 1;

    return std::make_tuple(extinction_factor, beta, delete_probability);
  }
  case CHAMFER: {
    const float extinction_factor = 0.23f * exp(-0.0082 * viewing_dist);
    const float beta = (-0.006f * viewing_dist) + 2.31;
    const float delete_probability = -0.7f * exp(-0.024 * viewing_dist) + 1;

    return std::make_tuple(extinction_factor, beta, delete_probability);
  }
  default:
    // NOTE(tom): The switch case should be exhaustive, so this statement
    //            should never be reached!
    assert(false);
  }
}

[[nodiscard]] std::optional<std::vector<torch::Tensor>>
fog(const torch::Tensor &point_cloud, const float prob,
    const fog_parameter metric, const float sigma, const int mean) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);

  if (const auto rand = distrib(rng); prob > rand) {

    const auto viewing_dist = get_truncated_normal_value(mean, sigma, 10, mean);

    const dimensions pc_dims = {point_cloud.size(0), point_cloud.size(1),
                                point_cloud.size(2)};

    std::vector<torch::Tensor> batch;
    batch.reserve(static_cast<std::size_t>(pc_dims.batch_size));
    for (tensor_size_t i = 0; i < pc_dims.batch_size; i++) {
      auto new_pc = fog(point_cloud.index({i}), metric, viewing_dist);

      batch.emplace_back(new_pc);
    }

    return batch;
  } else {
    // NOTE(tom): prob <= rand
    return std::nullopt;
  }
}

[[nodiscard]] torch::Tensor
fog(torch::Tensor point_cloud, const fog_parameter metric,
    const float viewing_dist, point_cloud_data::intensity_range max_intensity) {

  const auto [extinction_factor, beta, delete_probability] =
      calculate_factors(metric, viewing_dist);

  // selecting points for modification and deletion
  const auto dist = torch::sqrt(torch::sum(
      torch::pow(point_cloud.index({Slice(), Slice(None, 3)}), 2), 1));

  const auto modify_probability = 1 - torch::exp(-extinction_factor * dist);
  const auto modify_threshold = torch::rand(modify_probability.size(0));

  const auto selected = modify_threshold < modify_probability;

  const auto delete_threshold = torch::rand(point_cloud.size(0));
  const auto deleted =
      torch::logical_and(delete_threshold < delete_probability, selected);

  // changing intensity of unaltered points according to beer lambert law
  point_cloud.index({torch::logical_not(selected), 3}) *=
      torch::exp(-(2.99573 / viewing_dist) * 2 *
                 dist.index({torch::logical_not(selected)}));

  // changing position and intensity of selected points
  const auto altered_points =
      torch::logical_and(selected, torch::logical_not(deleted));

  if (const tensor_size_t num_altered_points =
          point_cloud.index({altered_points, Slice(None, 3)}).size(0);
      num_altered_points > 0) {
    const auto newdist =
        torch::empty(num_altered_points).exponential_(1 / beta) + 1.3;

    point_cloud.index_put_(
        {altered_points, Slice(None, 3)},
        point_cloud.index({altered_points, Slice(None, 3)}) *
            torch::reshape(newdist / dist.index({altered_points}), {-1, 1}));

    point_cloud.index_put_(
        {altered_points, 3},
        torch::empty({num_altered_points})
            .uniform_(0, static_cast<float>(max_intensity) * 0.3));
  }

  // delete points
  return point_cloud.index({torch::logical_not(deleted), Slice()});
}

[[nodiscard]] torch::Tensor
rain(torch::Tensor point_cloud,
     const cpp_utils::distribution_ranges<float> &dims,
     const uint32_t num_drops, const float precipitation, const distribution d,
     const point_cloud_data::intensity_range max_intensity) {

  point_cloud_data::max_intensity::set(max_intensity);

  auto [nf, si] = rt::generate_noise_filter<float, F32>(dims, num_drops,
                                                        precipitation, 1, d);

  return rt::trace(point_cloud, nf, si, simulation_type::rain, 0.9);
}

[[nodiscard]] std::optional<torch::Tensor>
rain(torch::Tensor point_cloud, const std::string_view noise_filter_path,
     const uint32_t num_drops_sigma, const float precipitation_sigma,
     const float prob) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);

  if (const auto rand = distrib(rng); prob > rand) {
    const auto r = static_cast<std::int32_t>(std::floor(
        get_truncated_normal_value(0, precipitation_sigma, 0, 20) + 1));
    const auto n =
        static_cast<std::int32_t>(std::floor(
            get_truncated_normal_value(0, num_drops_sigma, 0, 6) + 1)) *
        200;

    const auto filename =
        "nf_N=" + std::to_string(n) + "_R=" + std::to_string(r) + ".npz";
    std::filesystem::path noise_file(noise_filter_path);
    noise_file.append(filename);

    auto npz_data = cnpy::npz_load(noise_file);

    auto nf_array = npz_data["nf"];
    const auto nf =
        torch::from_blob(nf_array.data<float>(),
                         {static_cast<tensor_size_t>(nf_array.num_vals())})
            .reshape({-1, 6});

    auto si_array = npz_data["si"];
    const auto si =
        torch::from_blob(si_array.data<tensor_size_t>(),
                         {static_cast<tensor_size_t>(si_array.num_vals())});
    const auto result = rt::trace(point_cloud, nf, si, simulation_type::rain);
    point_cloud = result;
    return result;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<torch::Tensor>
snow(torch::Tensor point_cloud, const std::string_view noise_filter_path,
     const uint32_t num_drops_sigma, const float precipitation_sigma,
     const int32_t scale, const float prob) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);

  if (const auto rand = distrib(rng); prob > rand) {
    const auto r = static_cast<std::int32_t>(std::floor(
        get_truncated_normal_value(0, precipitation_sigma, 0, 10) + 1));
    const auto n =
        static_cast<std::int32_t>(std::floor(
            get_truncated_normal_value(0, num_drops_sigma, 0, 12) + 1)) *
        100;

    const auto filename =
        "nf_N=" + std::to_string(n) + "_R=" + std::to_string(r) + ".npz";
    std::filesystem::path noise_file(noise_filter_path);
    noise_file.append(filename);
    auto npz_data = cnpy::npz_load(noise_file);

    auto nf_array = npz_data["nf"];
    const auto nf =
        torch::from_blob(nf_array.data<float>(),
                         {static_cast<tensor_size_t>(nf_array.num_vals())})
            .reshape({-1, 6});

    nf.index({Slice(), 4}) *= scale;

    auto si_array = npz_data["si"];
    const auto si =
        torch::from_blob(si_array.data<tensor_size_t>(),
                         {static_cast<tensor_size_t>(si_array.num_vals())});

    auto result = rt::trace(point_cloud, nf, si, simulation_type::snow, 1.25);
    result.index_put_({result.index({Slice(), 3}) > 255, 3}, 255);

    point_cloud = result;
    return result;
  }

  return std::nullopt;
}

[[nodiscard]] torch::Tensor
snow(torch::Tensor point_cloud,
     const cpp_utils::distribution_ranges<float> &dims,
     const uint32_t num_drops, const float precipitation, const int32_t scale,
     point_cloud_data::intensity_range max_intensity) {

  point_cloud_data::max_intensity::set(max_intensity);

  auto [nf, si] = rt::generate_noise_filter<float, F32>(
      dims, num_drops, precipitation, scale, distribution::gm);

  point_cloud = rt::trace(point_cloud, nf, si, simulation_type::snow, 1.25);

  point_cloud.index(
      {point_cloud.index({Slice(), 3}) > static_cast<float>(max_intensity),
       3}) = static_cast<float>(max_intensity);

  return point_cloud;
}

void universal_weather(torch::Tensor point_cloud, const float prob,
                       const float sigma, const int mean, const float ext_a,
                       const float ext_b, const float beta_a,
                       const float beta_b, const float del_a, const float del_b,
                       const int int_a, const int int_b, const int mean_int,
                       const int int_range) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);

  if (const auto rand = distrib(rng); prob > rand) {
    const auto viewing_dist = get_truncated_normal_value(mean, sigma, 0, mean);

    const auto extinction_factor = ext_a * exp(ext_b * viewing_dist);

    const auto beta = (-beta_a * viewing_dist) + beta_b;
    const auto delete_probability = -del_a * exp(-del_b * viewing_dist) + 1;

    // selecting points for modification and deletion
    const auto dist = torch::sqrt(
        torch::sum(point_cloud.index({Slice(), Slice(None, 3)}).pow(2), 1));
    const auto modify_probability = 1 - torch::exp(-extinction_factor * dist);
    const auto modify_threshold = torch::rand(modify_probability.size(0));
    const auto selected = modify_threshold < modify_probability;
    const auto delete_threshold = torch::rand(point_cloud.size(0));
    const auto deleted =
        torch::logical_and(delete_threshold < delete_probability, selected);

    // changing intensity of unaltered points according to parametrized beer
    // lambert law
    point_cloud.index({torch::logical_not(selected), 3}) *=
        int_a * torch::exp(-(int_b / viewing_dist) *
                           dist.index({torch::logical_not(selected)}));

    // changing position and intensity of selected points
    const auto altered_points =
        torch::logical_and(selected, torch::logical_not(deleted));
    if (const auto num_altered_points =
            point_cloud.index({altered_points, Slice(None, 3)}).size(0);
        num_altered_points > 0) {

      const auto newdist =
          torch::empty(num_altered_points).exponential_(1 / beta) + 1.3;

      point_cloud.index_put_(
          {altered_points, Slice(None, 3)},
          point_cloud.index({altered_points, Slice(None, 3)}) *
              torch::reshape(newdist / dist.index({altered_points}), {-1, 1}));

      const auto min_int = std::max(mean_int - (int_range / 2), 0);
      const auto max_int = std::min(mean_int + (int_range / 2), 255);

      point_cloud.index_put_(
          {altered_points, 3},
          torch::empty({num_altered_points}).uniform_(min_int, max_int));
    }

    // delete points
    point_cloud = point_cloud.index({torch::logical_not(deleted), Slice()});
    point_cloud.index({point_cloud.index({Slice(), 3}) > 255, 3}) = 255;
  }
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "../include/weather_bindings.hpp"
#endif

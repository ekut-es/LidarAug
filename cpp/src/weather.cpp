
#include "../include/weather.hpp"
#include "../include/stats.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/pow.h>
#include <cstddef>
#include <random>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <tuple>

using namespace torch::indexing;
using namespace torch_utils;

[[nodiscard]] inline std::tuple<float, float, float>
calculate_factors(fog_parameter metric, float viewing_dist) {
  switch (metric) {
  case DIST: {
    const float extinction_factor = 0.32 * exp(-0.022 * viewing_dist);
    const float beta = (-0.00846 * viewing_dist) + 2.29;
    const float delete_probability = -0.63 * exp(-0.02 * viewing_dist) + 1;

    return std::make_tuple(extinction_factor, beta, delete_probability);
  }
  case CHAMFER: {
    const float extinction_factor = 0.23 * exp(-0.0082 * viewing_dist);
    const float beta = (-0.006 * viewing_dist) + 2.31;
    const float delete_probability = -0.7 * exp(-0.024 * viewing_dist) + 1;

    return std::make_tuple(extinction_factor, beta, delete_probability);
  }
  default:
    // NOTE(tom): The switch case should be exhaustive, so this statement
    //            should never be reached!
    assert(false);
  }
}

[[nodiscard]] std::optional<std::vector<torch::Tensor>>
fog(const torch::Tensor &point_cloud, const float prob, fog_parameter metric,
    const float sigma, const int mean) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);
  auto rand = distrib(rng);

  if (prob > rand) {

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

[[nodiscard]] torch::Tensor fog(torch::Tensor point_cloud, fog_parameter metric,
                                float viewing_dist, float max_intensity) {

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
  const tensor_size_t num_altered_points =
      point_cloud.index({altered_points, Slice(None, 3)}).size(0);

  if (num_altered_points > 0) {
    auto newdist =
        torch::empty(num_altered_points).exponential_(1 / beta) + 1.3;

    point_cloud.index_put_(
        {altered_points, Slice(None, 3)},
        point_cloud.index({altered_points, Slice(None, 3)}) *
            torch::reshape(newdist / dist.index({altered_points}), {-1, 1}));

    point_cloud.index_put_(
        {altered_points, 3},
        torch::empty({num_altered_points}).uniform_(0, max_intensity * 0.3));
  }

  // delete points
  return point_cloud.index({torch::logical_not(deleted), Slice()});
}

[[nodiscard]] torch::Tensor
rain(torch::Tensor point_cloud, std::array<float, 6> dims, uint32_t num_drops,
     float precipitation, distribution d,
     point_cloud_data::intensity_range max_intensity) {

  point_cloud_data::max_intensity::set(max_intensity);

  auto [nf, si] = rt::generate_noise_filter<float, F32>(dims, num_drops,
                                                        precipitation, 1, d);

  return rt::trace(point_cloud, nf, si, simulation_type::rain, 0.9);
}

[[nodiscard]] torch::Tensor
snow(torch::Tensor point_cloud, std::array<float, 6> dims, uint32_t num_drops,
     float precipitation, int32_t scale,
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

void universal_weather(torch::Tensor point_cloud, float prob, float sigma,
                       int mean, float ext_a, float ext_b, float beta_a,
                       float beta_b, float del_a, float del_b, int int_a,
                       int int_b, int mean_int, int int_range) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);
  auto rand = distrib(rng);

  if (prob > rand) {
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
    const auto num_altered_points =
        point_cloud.index({altered_points, Slice(None, 3)}).size(0);
    if (num_altered_points > 0) {

      auto newdist =
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

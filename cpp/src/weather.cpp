
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

[[nodiscard]] inline torch::Tensor
select_points(const torch::Tensor &point_cloud, tensor_size_t num_items,
              float viewing_dist, float extinction_factor,
              float delete_probability, float beta,
              std::uniform_real_distribution<float> &percentage_distrib) {

  const auto dist =
      point_cloud.index({Slice(), Slice(None, 3)}).pow(2).sum(1).sqrt();
  const auto modify_probability = 1 - exp(-extinction_factor * dist);

  const auto threshold_data_size = modify_probability.size(0);

  const auto modify_threshold =
      draw_values<float, F32>(percentage_distrib, threshold_data_size);

  const auto selected = modify_threshold < modify_probability;

  const auto delete_threshold =
      draw_values<float, F32>(percentage_distrib, num_items);

  const auto deleted =
      selected.logical_and(delete_threshold < delete_probability);

  // changing intensity of unaltered points according to beer lambert law
  point_cloud.index({selected.logical_not(), 3}) *=
      exp(-(2.99573 / viewing_dist) * 2 * dist.index({selected.logical_not()}));

  const auto altered_points = selected.logical_and(deleted.logical_not());
  const auto num_altered_points =
      point_cloud.index({altered_points, Slice(None, 3)}).size(0);

  if (num_altered_points > 0) {
    std::exponential_distribution<float> exp_d(beta);
    const auto new_dist =
        draw_values<float, F32>(exp_d, num_altered_points) + 1.3;

    point_cloud.index({altered_points, Slice(None, 3)}) *=
        (new_dist / dist.index({altered_points})).reshape({-1, 1});
  }

  std::uniform_real_distribution<float> d(0, 82);

  point_cloud.index({altered_points, 3}) =
      draw_values<float, F32>(d, num_altered_points);

  return point_cloud.index({deleted.logical_not(), Slice()});
}

[[nodiscard]] std::optional<std::vector<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_parameter metric,
    float sigma, int mean) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);
  auto rand = distrib(rng);

  if (prob > rand) {

    const auto viewing_dist = get_truncated_normal_value(mean, sigma, 10, mean);

    const auto [extinction_factor, beta, delete_probability] =
        calculate_factors(metric, viewing_dist);

    const dimensions pc_dims = {point_cloud.size(0), point_cloud.size(1),
                                point_cloud.size(2)};

    std::vector<torch::Tensor> batch;
    batch.reserve(static_cast<std::size_t>(pc_dims.batch_size));
    for (tensor_size_t i = 0; i < pc_dims.batch_size; i++) {
      auto new_pc =
          select_points(point_cloud.index({i}), pc_dims.num_items, viewing_dist,
                        extinction_factor, delete_probability, beta, distrib);

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
    point_cloud.index({altered_points, Slice(None, 3)}) *=
        torch::reshape(newdist / dist.index({altered_points}), {-1, 1});
    point_cloud.index({altered_points, 3}) =
        torch::empty({num_altered_points}).uniform_(0, max_intensity * 0.3);
  }

  // delete points
  return point_cloud.index({torch::logical_not(deleted), Slice()});
}

[[nodiscard]] torch::Tensor rain(torch::Tensor point_cloud,
                                 std::array<float, 6> dims, uint32_t num_drops,
                                 float precipitation, distribution d) {

  auto [nf, si] =
      rt::generate_noise_filter(dims, num_drops, precipitation, 1, d);
  return rt::trace(point_cloud, nf, si, 0.9);
}

[[nodiscard]] torch::Tensor snow(torch::Tensor point_cloud,
                                 std::array<float, 6> dims, uint32_t num_drops,
                                 float precipitation, int32_t scale,
                                 float max_intensity) {

  auto [nf, si] =
      rt::generate_noise_filter(dims, num_drops, precipitation, scale, GM);
  point_cloud = rt::trace(point_cloud, nf, si, 1.25);
  point_cloud.index({point_cloud.index({Slice(), 3}) > max_intensity, 3}) =
      max_intensity;
  return point_cloud;
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "../include/weather_bindings.hpp"
#endif

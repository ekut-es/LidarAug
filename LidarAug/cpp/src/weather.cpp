
#include "../include/weather.hpp"
#include "../include/stats.hpp"
#include "../include/tensor.hpp"
#include <ATen/ops/from_blob.h>
#include <ATen/ops/pow.h>
#include <cstddef>
#include <random>
#include <torch/csrc/autograd/generated/variable_factories.h>

using namespace torch::indexing;

[[nodiscard]] inline torch::Tensor
select_points(const torch::Tensor &point_cloud, tensor_size_t num_items,
              float viewing_dist, float extinction_factor,
              float delete_probability, float beta,
              std::uniform_real_distribution<float> &percentage_distrib) {

  const auto dist = point_cloud.index({Slice(), Slice(None, 3)}).pow(2).sqrt();
  const auto modify_probability = 1 - exp(-extinction_factor * dist);

  const auto threshold_data_size = modify_probability.size(0);

  // NOTE(tom): `from_blob` only provides a view into the vector, so the
  //            vector has to outlive the tensor
  const auto modify_threshold =
      torch::from_blob(std::get<VECTOR>(draw_values<float>(percentage_distrib,
                                                           threshold_data_size))
                           .data(),
                       {threshold_data_size}, torch::kF32);

  const auto selected = modify_threshold < modify_probability;

  // NOTE(tom): `from_blob` only provides a view into the vector, so the
  //            vector has to outlive the tensor
  const auto delete_threshold = torch::from_blob(
      std::get<VECTOR>(draw_values<float>(percentage_distrib, num_items))
          .data(),
      {num_items}, torch::kF32);

  const auto deleted =
      selected.logical_and(delete_threshold < delete_probability);

  // changing intensity of unaltered points according to beer lambert law
  point_cloud.index({selected.logical_not(), 3}) *=
      exp(-(2.99573 / viewing_dist) * 2 * dist[selected.logical_not()]);

  const auto altered_points = selected.logical_and(deleted.logical_not());
  const auto num_altered_points =
      point_cloud.index({altered_points, Slice(None, 3)}).item<tensor_size_t>();

  if (num_altered_points > 0) {
    std::exponential_distribution<float> exp_d(beta);
    const auto new_dist =
        torch::from_blob(
            std::get<VECTOR>(draw_values<float>(exp_d, num_altered_points))
                .data(),
            {num_altered_points}, torch::kF32) +
        1.3;
    point_cloud.index({altered_points, Slice(None, 3)}) *=
        (new_dist / dist.index({altered_points})).reshape({-1, 1});
  }

  std::uniform_real_distribution<float> d(0, 82);

  // TODO(tom): This needs review!
  point_cloud.index({altered_points, 3}) = torch::from_blob(
      std::get<VECTOR>(
          draw_values<float>(d, static_cast<std::size_t>(num_altered_points)))
          .data(),
      {num_altered_points}, torch::kF32);

  return point_cloud.index({deleted.logical_not(), Slice()});
}

[[nodiscard]] std::optional<torch::List<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_metric metric,
    float sigma, int mean) {

  auto rng = get_rng();
  std::uniform_real_distribution<float> distrib(0, HUNDRED_PERCENT - 1);
  auto rand = distrib(rng);

  if (prob > rand) {

    auto viewing_dist = get_truncated_normal_value(mean, sigma, 10, mean);

    const auto calculate_factors =
        [metric, viewing_dist]() -> std::tuple<float, float, float> {
      switch (metric) {
      case DIST: {
        const float extinction_factor = 0.32 * exp(-0.022 * viewing_dist);
        const float beta = (-0.00846 * viewing_dist) + 2.29;
        const float delete_probability = -0.63 * exp(-0.02 * viewing_dist) + 1;
        return {extinction_factor, beta, delete_probability};
      }
      case CHAMFER: {
        const float extinction_factor = 0.23 * exp(-0.0082 * viewing_dist);
        const float beta = (-0.006 * viewing_dist) + 2.31;
        const float delete_probability = -0.7 * exp(-0.024 * viewing_dist) + 1;
        return {extinction_factor, beta, delete_probability};
      }
      }
    };

    const auto [extinction_factor, beta, delete_probability] =
        calculate_factors();

    const dimensions pc_dims = {point_cloud.size(0), point_cloud.size(1),
                                point_cloud.size(2)};

    torch::List<torch::Tensor> batch;
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

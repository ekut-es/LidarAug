
#include "../include/raytracing.hpp"
#include "../include/stats.hpp"
#include "../include/weather.hpp"
#include <ATen/TensorIndexing.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

using namespace torch_utils;
using Slice = torch::indexing::Slice;

// Set minimum intersection distance to 1m
constexpr tensor_size_t min_intersect_dist = 1;

[[nodiscard]] torch::Tensor rt::trace(torch::Tensor point_cloud,
                                      const torch::Tensor &noise_filter,
                                      const torch::Tensor &split_index,
                                      const simulation_type sim_t,
                                      const float intensity_factor /*= 0.9*/) {

  const auto num_points = point_cloud.size(0);
  constexpr auto num_rays = 11;

  const auto intersections = torch::zeros({num_points, num_rays}, F32);
  const auto distances = torch::zeros({num_points, num_rays}, F32);
  const auto distance_count = torch::zeros({num_points, num_rays}, I64);
  const auto most_intersect_count = torch::zeros({num_points}, I64);
  const auto most_intersect_dist = torch::zeros({num_points}, F32);

  rt::intersects(point_cloud, noise_filter, split_index, intersections,
                 distances, distance_count, most_intersect_count,
                 most_intersect_dist, num_points, sim_t, intensity_factor);

  // select all points where any of x, y, z != 0
  const auto indices =
      (point_cloud.index({Slice(), Slice(0, 3)}) != 0).sum(1).nonzero();

  auto result = point_cloud.index({indices}).squeeze(1);

  return result;
}

[[nodiscard]] float rt::trace_beam(const torch::Tensor &noise_filter,
                                   const torch::Tensor &beam,
                                   const torch::Tensor &split_index) {

  const auto beam_length = rt::vector_length(beam);
  const auto beam_normalized = rt::normalize(beam);

  const auto index =
      static_cast<int>(((torch::atan2(beam[1], beam[0]).item<float>() * 180 /
                         math_utils::PI_RAD) +
                        360) *
                       nf_split_factor) %
      (360 * nf_split_factor);

  for (auto i = split_index[index].item<tensor_size_t>();
       i < split_index[index + 1].item<tensor_size_t>(); i++) {
    const auto nf = noise_filter[i];

    const auto sphere = torch::tensor(
        {nf[0].item<float>(), nf[1].item<float>(), nf[2].item<float>()});

    if (beam_length < nf[3].item<float>()) {
      return -1;
    }

    if (const auto length_beam_sphere = rt::scalar(sphere, beam_normalized);
        length_beam_sphere > 0.0) {

      if (const auto dist_beam_sphere =
              sqrt(nf[3].item<float>() * nf[3].item<float>() -
                   length_beam_sphere * length_beam_sphere);
          dist_beam_sphere < nf[4].item<float>()) {

        return nf[3].item<float>();
      }
    }
  }

  return -1;
}

void rt::intersects(torch::Tensor point_cloud,
                    const torch::Tensor &noise_filter,
                    const torch::Tensor &split_index,
                    torch::Tensor intersections, torch::Tensor distances,
                    torch::Tensor distance_count,
                    torch::Tensor most_intersect_count,
                    torch::Tensor most_intersect_dist,
                    const tensor_size_t num_points, const simulation_type sim_t,
                    const float intensity_factor) {

  constexpr auto num_rays = 11;

  const auto t = r_table.at(static_cast<size_t>(sim_t));

  // for som reason I can't use structured bindings here with clang
  const auto r_all_threshold = std::get<0>(t);
  const auto r_most_threshold = std::get<1>(t);

#pragma omp parallel for schedule(dynamic)
  for (tensor_size_t i = 0; i < num_points; i++) {

    const auto original_point = point_cloud.index({i, Slice(0, 3)});
    auto beam = point_cloud.index({i, Slice(0, 3)});

    tensor_size_t idx_count = 0;

    // --- get original intersection ---
    auto intersection_dist = rt::trace_beam(noise_filter, beam, split_index);
    if (intersection_dist > 0) {
      intersections.index_put_({i, idx_count}, intersection_dist);
      idx_count += 1;
    }

    // --- rotate points ---
    constexpr auto num_points_per_streak = 2;
    constexpr auto divergence_angle = 2e-4;
    constexpr auto vector_rotation_angle = M_PI / 5;
    constexpr auto num_streaks = 5;
    const auto z_axis = torch::tensor({0.0, 0.0, 1.0});

    auto rot_vec = rt::normalize(rt::cross(beam, z_axis));

    for (auto j = 0; j < num_streaks; j++) {
      for (auto k = 1; k < num_points_per_streak + 1; k++) {

        beam = rt::rotate(original_point, rot_vec,
                          (k <= num_points_per_streak / 2)
                              ? k * divergence_angle
                              : (k - (num_points_per_streak / 2.0f)) *
                                    (-divergence_angle));

        intersection_dist = rt::trace_beam(noise_filter, beam, split_index);

        if (intersection_dist > min_intersect_dist) {
          intersections.index_put_({i, idx_count}, intersection_dist);
          idx_count += 1;
        } else if (/* min_intersect_dist > */ intersection_dist > 0) {
          intersections.index_put_({i, idx_count}, 1);
          idx_count += 1;
        }
        rot_vec = rt::rotate(rot_vec, rt::normalize(original_point),
                             vector_rotation_angle);
      }
    }

    // --- count intersections ---
    uint32_t n_intersects = 0;

    for (auto ii = 0; ii < intersections.size(1); ii++) {
      const auto intersect = intersections[i][ii].item<float>();
      if (intersect != 0)
        n_intersects += 1;
      for (tensor_size_t j = 0; j < num_rays; j++) {
        if (intersect != 0) {
          if (distances[i][j].item<float>() == 0) {
            distance_count.index_put_({i, j}, 1);
            distances.index_put_({i, j}, intersect);
            break;
          }

          else if (intersect == distances[i][j].item<float>()) {
            distance_count[i][j] += 1;
            break;
          }
        }
      }
    }

    // --- find most intersected drop ---
    tensor_size_t max_count = 0;
    auto max_intersection_dist = 0.0;

    for (auto ii = 0; ii < distance_count.size(1); ii++) {
      if (const auto count = distance_count[i][ii].item<tensor_size_t>();
          count > max_count) {
        max_count = count;
        max_intersection_dist = distances[i][ii].item<float>();
      }
      most_intersect_count.index_put_({i}, max_count);
      most_intersect_dist.index_put_({i}, max_intersection_dist);
    }

    if (const auto r_all = n_intersects / static_cast<double>(num_rays);
        r_all > r_all_threshold) {
      assert(n_intersects != 0);
      if (const auto r_most = max_count / static_cast<double>(n_intersects);
          r_most > r_most_threshold) { // set point towards sensor

        const auto dist = rt::vector_length(point_cloud[i]);

        point_cloud.index({i, Slice(0, 3)}) *= max_intersection_dist / dist;
        point_cloud.index_put_({i, 3}, get_intensity(sim_t));
      } else { // delete point (filtered out later)
        point_cloud.index_put_({i, Slice(0, 4)}, 0);
      }
    } else { // modify intensity of unaltered point
      point_cloud[i][3] *= intensity_factor;
    }
  }
}

[[nodiscard]] torch::Tensor rt::sample_particles(int64_t num_particles,
                                                 const float precipitation,
                                                 const distribution d) {
  constexpr std::array function_table = {
      inverted_exponential_cdf,
      inverted_lognormal_cdf,
      inverted_exponential_gm,
  };

  const auto f = function_table.at(static_cast<size_t>(d));

  return f(torch::rand({num_particles}), precipitation) * (1 / 2000.0);
}

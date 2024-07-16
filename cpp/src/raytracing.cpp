
#include "../include/raytracing.hpp"
#include "../include/stats.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/clone.h>
#include <c10/core/TensorOptions.h>
#include <cstdio>
#include <omp.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

using namespace torch_utils;
using Slice = torch::indexing::Slice;

constexpr auto nf_split_factor = 32;

[[nodiscard]] torch::Tensor rt::trace(torch::Tensor point_cloud,
                                      const torch::Tensor &noise_filter,
                                      const torch::Tensor &split_index,
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
                 most_intersect_dist, num_points, intensity_factor);

  // select all points where any of x, y, z != 0
  const auto indices =
      (point_cloud.index({Slice(), Slice(0, 3)}) != 0).sum(1).nonzero();

  auto result = point_cloud.index({indices}).squeeze(1);

  return result;
}

[[nodiscard]] float rt::trace_beam(const torch::Tensor &noise_filter,
                                   const torch::Tensor &beam,
                                   const torch::Tensor &split_index) {

  const auto index =
      static_cast<int>(
          ((atan2(beam[1].item<float>(), beam[0].item<float>()) * 180 / M_PI) +
           360) *
          nf_split_factor) %
      (360 * nf_split_factor);

  for (auto i = split_index[index].item<tensor_size_t>();
       i < split_index[index + 1].item<tensor_size_t>(); i++) {
    const auto nf = noise_filter[i];

    const auto sphere = torch::tensor(
        {nf[0].item<float>(), nf[1].item<float>(), nf[2].item<float>()});

    if (const auto beam_dist = rt::vector_length(beam);
        beam_dist < nf[3].item<float>())
      return -1;

    if (const auto length_beam_sphere = rt::scalar(sphere, rt::normalize(beam));
        length_beam_sphere > 0.0) {

      if (const auto dist_beam_sphere =
              sqrt(nf[3].item<float>() * nf[3].item<float>() -
                   length_beam_sphere * length_beam_sphere);
          dist_beam_sphere < nf[4].item<float>())

        return nf[3].item<float>();
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
                    const tensor_size_t num_points, float intensity_factor) {

  constexpr auto num_rays = 11;

  const auto get_original_intersection = [&intersections, noise_filter](
                                             const torch::Tensor &beam,
                                             const torch::Tensor &split_index,
                                             const tensor_size_t index) {
    tensor_size_t idx_count = 0;

    auto intersection_dist = rt::trace_beam(noise_filter, beam, split_index);
    if (intersection_dist > 0) {
      intersections.index_put_({index, idx_count}, intersection_dist);
      idx_count += 1;
    }

    return std::make_pair(intersection_dist, idx_count);
  };

  const auto rotate_points = [noise_filter, &intersections,
                              split_index](const torch::Tensor &original_point,
                                           float &intersection_dist,
                                           tensor_size_t &idx_count,
                                           tensor_size_t index) {
    constexpr auto vector_rotation_angle = M_PI / 5;
    constexpr auto num_streaks = 5;
    const auto z_axis = torch::tensor({0.0, 0.0, 1.0});

    auto rot_vec = normalize(rt::cross(original_point, z_axis));

    for (auto j = 0; j < num_streaks; j++) {
      constexpr auto num_points_per_streak = 2;
      for (auto k = 1; k < num_points_per_streak + 1; k++) {
        constexpr auto divergence_angle = 2e-4;

        auto beam = rt::rotate(original_point, rot_vec,
                               (k <= num_points_per_streak / 2)
                                   ? k * divergence_angle
                                   : (k - (num_points_per_streak / 2.0f)) *
                                         (-divergence_angle));

        intersection_dist = rt::trace_beam(noise_filter, beam, split_index);

        if (intersection_dist > 0) {
          intersections.index_put_({index, idx_count}, intersection_dist);
          idx_count += 1;
        }
        rot_vec = rt::rotate(rot_vec, rt::normalize(original_point),
                             vector_rotation_angle);
      }
    }
  };

  const auto count_intersections = [intersections, &distances,
                                    &distance_count](tensor_size_t index) {
    uint32_t n_intersects = 0;

    for (auto i = 0; i < intersections.size(1); i++) {
      const auto intersect = intersections[index][i].item<float>();
      if (intersect != 0)
        n_intersects += 1;
      for (tensor_size_t j = 0; j < num_rays; j++) {
        if (intersect != 0) {
          if (distances[index][j].item<float>() == 0) {
            distance_count.index_put_({index, j}, 1);
            distances.index_put_({index, j}, intersect);
            break;
          }

          if (intersect == distances[index][j].item<float>()) {
            distance_count[index][j] += 1;
            break;
          }
        }
      }
    }

    return n_intersects;
  };

  const auto find_most_intersected_drop =
      [&most_intersect_count, &most_intersect_dist, distances, &point_cloud,
       intensity_factor](const torch::Tensor &distance_count,
                         const uint32_t n_intersects,
                         const tensor_size_t index) {
        tensor_size_t max_count = 0;
        auto max_intersection_dist = 0.0;

        for (auto i = 0; i < distance_count.size(1); i++) {
          if (const auto count = distance_count[index][i].item<tensor_size_t>();
              count > max_count) {
            max_count = count;
            max_intersection_dist = distances[index][i].item<float>();
          }
          most_intersect_count.index_put_({index}, max_count);
          most_intersect_dist.index_put_({index}, max_intersection_dist);
        }

        if (const auto r_all = n_intersects / static_cast<double>(num_rays);
            r_all > 0.15) {
          assert(n_intersects != 0);
          if (const auto r_most = max_count / static_cast<double>(n_intersects);
              r_most > 0.8) { // set point towards sensor

            const auto dist = rt::vector_length(point_cloud[index]);

            point_cloud.index({index, Slice(0, 3)}) *=
                max_intersection_dist / dist;
            point_cloud[index][3] *= 0.005;
          } else { // delete point (filtered out later)
            point_cloud.index_put_({index, Slice(0, 4)}, 0);
          }
        } else { // modify intensity of unaltered point
          point_cloud[index][3] *= intensity_factor;
        }
      };

#pragma omp parallel for schedule(dynamic)
  for (tensor_size_t i = 0; i < num_points; i++) {

    const auto original_point = point_cloud.index({i, Slice(0, 3)});

    auto [intersection_dist, idx_count] =
        get_original_intersection(original_point, split_index, i);

    rotate_points(original_point, intersection_dist, idx_count, i);

    const auto n_intersects = count_intersections(i);

    find_most_intersected_drop(distance_count, n_intersects, i);
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

  return f(torch::rand({num_particles}), precipitation) * (1 / 2000);
}

// TODO(tom): Make dim a 'distribution_ranges' (found in transformations.hpp,
// needs to go in utils or something)
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> rt::generate_noise_filter(
    const std::array<float, 6> &dim, const uint32_t drops_per_m3,
    const float precipitation, const int32_t scale, const distribution d) {

  const auto total_drops = static_cast<int>(
      std::abs(dim[0] - dim[1]) * std::abs(dim[2] - dim[3]) *
      std::abs(dim[4] - dim[5]) * static_cast<float>(drops_per_m3));

  const auto x = torch::empty({total_drops}).uniform_(dim[0], dim[1]);
  const auto y = torch::empty({total_drops}).uniform_(dim[2], dim[3]);
  const auto z = torch::empty({total_drops}).uniform_(dim[4], dim[5]);

  const auto dist =
      torch::sqrt(torch::pow(x, 2) + torch::pow(y, 2) + torch::pow(z, 2));
  const auto size = sample_particles(total_drops, precipitation, d) * scale;

  const auto index =
      (((torch::arctan2(y, x) * 180 / math_utils::PI_RAD) + 360) *
       nf_split_factor)
          .toType(I32) %
      (360 * nf_split_factor);
  const auto nf = torch::stack({x, y, z, dist, size, index}, -1);

  return sort_noise_filter(nf);
}

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
rt::sort_noise_filter(torch::Tensor nf) {

  auto split_index = torch::zeros(360 * nf_split_factor + 1);

  nf = nf.index({nf.index({Slice(), 3}).argsort()});
  nf = nf.index({nf.index({Slice(), 5}).argsort()});

  if (!nf.is_contiguous()) {
    std::printf(
        "for performance reasons, please make sure that 'nf' is contiguous!");
    nf = torch::clone(nf, torch::MemoryFormat::Contiguous);
  }

  const auto *const nf_ptr = nf.const_data_ptr<float>();
  const auto row_size = nf.size(1);

#pragma omp parallel for
  for (tensor_size_t i = 0; i < nf.size(0) - 1; i++) {

    const auto idx = i * row_size + 5;

    const auto val_at_i = nf_ptr[idx];
    const auto val_at_i_plus_one = nf_ptr[idx + row_size];

    if (val_at_i != val_at_i_plus_one) {
      split_index.index_put_({static_cast<tensor_size_t>(val_at_i_plus_one)},
                             i + 1);
    }
  }

  split_index.index_put_({split_index.size(0) - 1}, nf.size(0) - 1);

  return std::make_pair(nf, split_index);
}

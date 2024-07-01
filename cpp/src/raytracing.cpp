
#include "../include/raytracing.hpp"
#include "../include/stats.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>
#include <utility>

using namespace torch_utils;
using Slice = torch::indexing::Slice;

#define NF_SPLIT_FACTOR 32

[[nodiscard]] torch::Tensor rt::trace(torch::Tensor point_cloud,
                                      const torch::Tensor &noise_filter,
                                      const torch::Tensor &split_index,
                                      const float intensity_factor /*= 0.9*/) {

  const auto num_points = point_cloud.size(0);
  constexpr auto num_rays = 11;

  const auto intersections = torch::zeros({num_points, num_rays}, F32);
  const auto distances = torch::zeros({num_points, num_rays}, F32);
  const auto distance_count = torch::zeros({num_points, num_rays}, F32);
  const auto most_intersect_count = torch::zeros({num_points}, F32);
  const auto most_intersect_dist = torch::zeros({num_points}, F32);

  // TODO(tom): Since this used to be CUDA code, it would probably be a
  //            nobrainer to make it multithreaded on the CPU as well
  rt::intersects(point_cloud, noise_filter, split_index, intersections,
                 distances, distance_count, most_intersect_count,
                 most_intersect_dist, num_points, intensity_factor);

  // select all points where x & y & z != 0
  const auto indices =
      (point_cloud.index({Slice(), Slice(0, 3)}) != 0).sum(1).nonzero();

  auto result = point_cloud.index({indices}).squeeze(1);

  return result;
}

[[nodiscard]] float rt::trace_beam(const torch::Tensor &noise_filter,
                                   const torch::Tensor &beam,
                                   const torch::Tensor &split_index) {

  // TODO(tom): this is very messy and needs revisiting

  const auto *const si = split_index.const_data_ptr<int>();
  const auto *const b = beam.const_data_ptr<float>();

  const auto index = static_cast<int>(((atan2(b[1], b[0]) * 180 / M_PI) + 360) *
                                      NF_SPLIT_FACTOR) %
                     (360 * NF_SPLIT_FACTOR);

  for (auto i = si[index]; i < si[index + 1]; i++) {
    const auto *const nf = noise_filter[i].const_data_ptr<float>();

    const auto sphere =
        (noise_filter[i][0], noise_filter[i][1], noise_filter[i][2]);
    const auto beam_dist = rt::vector_length(beam);
    if (beam_dist < nf[3])
      return -1;

    const auto length_beam_sphere = rt::scalar(sphere, rt::normalize(beam));
    if (length_beam_sphere > 0.0) {
      const auto dist_beam_sphere =
          sqrt((nf[3] * nf[3]) - (length_beam_sphere * length_beam_sphere));
      if (dist_beam_sphere < nf[4])
        return nf[3];
    }
  }

  return -1;
}

void rt::intersects(torch::Tensor point_cloud,
                    const torch::Tensor &noise_filter,
                    const torch::Tensor &split_index,
                    torch::Tensor intersections, const torch::Tensor &distances,
                    torch::Tensor distance_count,
                    torch::Tensor most_intersect_count,
                    torch::Tensor most_intersect_dist,
                    const tensor_size_t num_points, float intensity_factor) {

  constexpr auto NUM_RAYS = 11;

  const auto get_original_intersection = [intersections, noise_filter](
                                             const torch::Tensor &beam,
                                             const torch::Tensor &split_index,
                                             const tensor_size_t index) {
    tensor_size_t idx_count = 0;

    auto intersection_dist = rt::trace_beam(noise_filter, beam, split_index);
    if (intersection_dist > 0) {
      intersections[index][idx_count] = intersection_dist;
      idx_count += 1;
    }

    return std::make_pair(intersection_dist, idx_count);
  };

  const auto rotate_points = [noise_filter, intersections,
                              split_index](const torch::Tensor &original_point,
                                           float &intersection_dist,
                                           tensor_size_t &idx_count,
                                           tensor_size_t index) {
    constexpr auto divergence_angle = 2e-4;
    constexpr auto vector_rotation_angle = M_PI / 5;
    constexpr auto num_streaks = 5;
    constexpr auto num_points_per_streak = 2;
    const auto z_axis = torch::tensor({0.0, 0.0, 1.0});

    auto rot_vec = normalize(rt::cross(original_point, z_axis));

    for (auto j = 0; j < num_streaks; j++) {
      for (auto k = 1; k < num_points_per_streak + 1; k++) {

        auto beam = rt::rotate(original_point, rot_vec,
                               (k <= num_points_per_streak / 2)
                                   ? k * divergence_angle
                                   : (k - (num_points_per_streak / 2.0f)) *
                                         (-divergence_angle));

        intersection_dist = rt::trace_beam(noise_filter, beam, split_index);

        if (intersection_dist > 0) {
          intersections[index][idx_count] = intersection_dist;
          idx_count += 1;
        }
        rot_vec = rt::rotate(rot_vec, rt::normalize(original_point),
                             vector_rotation_angle);
      }
    }
  };

  const auto count_intersections = [intersections, distances,
                                    distance_count](tensor_size_t index) {
    uint32_t n_intersects = 0;

    for (auto i = 0; i < intersections.size(1); i++) {
      const auto intersect = intersections[index][i].item<float>();
      if (intersect != 0)
        n_intersects += 1;
      for (auto j = 0; j < NUM_RAYS; j++) {
        if (intersect != 0) {
          if (distances[index][j].item<float>() == 0) {
            distance_count[index][j] = 1;
            distances[index][j] = intersect;
            break;
          } else if (intersect == distances[index][j].item<float>()) {
            distance_count[index][j] += 1;
            break;
          }
        }
      }
    }

    return n_intersects;
  };

  const auto find_most_intersected_drop =
      [most_intersect_count, most_intersect_dist, distances, point_cloud,
       intensity_factor](const torch::Tensor &distance_count,
                         const uint32_t n_intersects,
                         const tensor_size_t index) {
        const auto r_all = n_intersects / NUM_RAYS;

        tensor_size_t max_count = 0;
        auto max_intersection_dist = 0.0;

        for (auto i = 0; i < distance_count.size(1); i++) {
          auto count = distance_count[index][i].item<tensor_size_t>();
          if (count > max_count) {
            max_count = count;
            max_intersection_dist = distances[index][i].item<float>();
          }
          most_intersect_count[index] = max_count;
          most_intersect_dist[index] = max_intersection_dist;

          const auto r_most = max_count / n_intersects;

          if (r_all > 0.15) {
            if (r_most > 0.8) { // set point towards sensor
              const auto dist = rt::vector_length(point_cloud[index]);

              point_cloud[index][0] *= max_intersection_dist / dist;
              point_cloud[index][1] *= max_intersection_dist / dist;
              point_cloud[index][2] *= max_intersection_dist / dist;
              point_cloud[index][3] *= 0.005;
            } else { // delete point (filtered out later)
              point_cloud[index][0] = 0;
              point_cloud[index][1] = 0;
              point_cloud[index][2] = 0;
              point_cloud[index][3] = 0;
            }
          } else { // modify intensity of unaltered point
            point_cloud[index][3] *= intensity_factor;
          }
        }
      };

  for (tensor_size_t i = 0; i < num_points; i++) {

    const auto original_point =
        (point_cloud[i][0], point_cloud[i][1], point_cloud[i][2]);

    auto [intersection_dist, idx_count] =
        get_original_intersection(original_point, split_index, i);

    rotate_points(original_point, intersection_dist, idx_count, i);

    const auto n_intersects = count_intersections(i);

    find_most_intersected_drop(distance_count, n_intersects, i);
  }
}

[[nodiscard]] torch::Tensor rt::sample_particles(int64_t num_particles,
                                                 const float precipitation,
                                                 distribution d) {
  torch::Tensor (*f)(torch::Tensor, float);

  switch (d) {
  case LOG_NORMAL:
    f = inverted_lognormal_cdf;
    break;
  case EXPONENTIAL:
    f = inverted_exponential_cdf;
    break;
  case GM:
    f = inverted_exponential_gm;
    break;
  }

  return f(torch::rand({num_particles}), precipitation) * (1 / 2000);
}

// TODO(tom): Make dim a 'distribution_ranges' (found in transformations.hpp,
// needs to go in utils or something)
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
rt::generate_noise_filter(std::array<float, 6> dim, uint32_t drops_per_m3,
                          const float precipitation, int32_t scale,
                          distribution d) {

  const auto total_drops =
      static_cast<int>(std::abs(dim[0] - dim[1]) * std::abs(dim[2] - dim[3]) *
                       std::abs(dim[4] - dim[5]) * drops_per_m3);
  // random.seed(42)

  std::uniform_real_distribution<float> x_ud(dim[0], dim[1]);
  std::uniform_real_distribution<float> y_ud(dim[2], dim[3]);
  std::uniform_real_distribution<float> z_ud(dim[4], dim[5]);

  const auto x = draw_values<float, F32>(x_ud, total_drops);
  const auto y = draw_values<float, F32>(y_ud, total_drops);
  const auto z = draw_values<float, F32>(z_ud, total_drops);

  const auto dist =
      torch::sqrt(torch::pow(x, 2) + torch::pow(y, 2) + torch::pow(z, 2));
  const auto size = sample_particles(total_drops, precipitation, d) * scale;

  const auto index =
      (((torch::arctan2(y, x) * 180 / math_utils::PI_RAD) + 360) *
       NF_SPLIT_FACTOR)
          .toType(I32) %
      (360 * NF_SPLIT_FACTOR);
  auto nf = torch::stack({x, y, z, dist, size, index}, -1);

  return sort_noise_filter(nf);
}

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
rt::sort_noise_filter(torch::Tensor nf) {

  auto split_index = torch::zeros(360 * NF_SPLIT_FACTOR + 1);

  nf = nf.index({nf.index({Slice(), 3}).argsort()});
  nf = nf.index({nf.index({Slice(), 5}).argsort()});

  for (tensor_size_t i = 0; i < nf.size(0) - 1; i++) {
    if (!nf.index({i, 5}).equal(nf.index({i + 1, 5}))) {

      split_index[nf.index({i + 1, 5}).item<tensor_size_t>()] = i + 1;
    }
  }
  split_index[split_index.size(0) - 1] = nf.size(0) - 1;

  return std::make_pair(nf, split_index);
}

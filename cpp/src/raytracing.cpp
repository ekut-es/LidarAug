
#include "../include/raytracing.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>

using namespace torch_utils;
using Slice = torch::indexing::Slice;

#define NF_SPLIT_FACTOR 32

[[nodiscard]] torch::Tensor
rt::trace(torch::Tensor point_cloud, const torch::Tensor &noise_filter,
          const torch::Tensor &split_index,
          std::optional<float> intensity_factor /*= 0.9*/) {

  const auto num_points = point_cloud.size(0);
  constexpr auto num_rays = 11;

  auto intersections = torch::zeros({num_points, num_rays}, F32);
  auto distances = torch::zeros({num_points, num_rays}, F32);
  auto distance_count = torch::zeros({num_points, num_rays}, F32);
  auto most_intersect_count = torch::zeros({num_points}, F32);
  auto most_intersect_dist = torch::zeros({num_points}, F32);

  // TODO(tom): Since this used to be CUDA code, it would probably be a
  //            nobrainer to make it multithreaded on the CPU as well
  rt::intersects(point_cloud, noise_filter, split_index, intersections,
                 distances, distance_count, most_intersect_count,
                 most_intersect_dist, num_points,
                 intensity_factor.value_or(0.9));

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

  const auto si = split_index.const_data_ptr<int>();
  const auto b = beam.const_data_ptr<float>();

  const auto index = static_cast<int>(((atan2(b[1], b[0]) * 180 / M_PI) + 360) *
                                      NF_SPLIT_FACTOR) %
                     (360 * NF_SPLIT_FACTOR);

  for (auto i = si[index]; i < si[index + 1]; i++) {
    const auto nf = noise_filter[i].const_data_ptr<float>();

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
                    torch::Tensor most_intersect_dist, tensor_size_t num_points,
                    float intensity_factor) {

  constexpr auto NUM_RAYS = 11;

  for (auto i = 0; i < num_points; i++) {

    auto original_point =
        (point_cloud[i][0], point_cloud[i][1], point_cloud[i][2]);
    auto idx_count = 0;

    // original point intersection
    auto intersection_dist =
        rt::trace_beam(noise_filter, original_point, split_index);
    if (intersection_dist > 0) {
      intersections[i][idx_count] = intersection_dist;
      idx_count += 1;
    }

    // ----------- rotated points ----------------
    auto divergence_angle = 2e-4;
    auto vector_rotation_angle = M_PI / 5;
    auto num_streaks = 5;
    auto num_points_per_streak = 2;
    auto z_axis = torch::tensor({0.0, 0.0, 1.0});
    auto rot_vec = normalize(rt::cross(original_point, z_axis));
    for (auto j = 0; j < num_streaks; j++) {

      for (auto k = 1; k < num_points_per_streak + 1; k++) {
        if (k <= num_points_per_streak / 2)
          original_point =
              rt::rotate(original_point, rot_vec, k * divergence_angle);
        else
          original_point = rt::rotate(original_point, rot_vec,
                                      (k - (num_points_per_streak / 2.0f)) *
                                          (-divergence_angle));

        intersection_dist =
            rt::trace_beam(noise_filter, original_point, split_index);

        if (intersection_dist > 0) {
          intersections[i][idx_count] = intersection_dist;
          idx_count += 1;
        }
        rot_vec = rt::rotate(rot_vec, rt::normalize(original_point),
                             vector_rotation_angle);
      }
    }

    // ------------ count intersections ------------
    auto n_intersects = 0;

    for (auto j = 0; j < intersections.size(1); j++) {
      auto intersect = intersections[i][j].item<float>();
      if (intersect != 0)
        n_intersects += 1;
      for (auto k = 0; k < NUM_RAYS; k++) {
        if (intersect != 0) {
          if (distances[i][k].item<float>() == 0) {
            distance_count[i][k] = 1;
            distances[i][k] = intersect;
            break;
          } else if (intersect == distances[i][k].item<float>()) {
            distance_count[i][k] += 1;
            break;
          }
        }
      }
    }

    auto r_all = n_intersects / NUM_RAYS;

    // --------- find most intersected drop -------

    auto max_count = 0;
    auto max_intersection_dist = 0.0;
    for (auto j = 0; j < distance_count.size(1); j++) {
      auto count = distance_count[i][j].item<tensor_size_t>();
      if (count > max_count) {
        max_count = count;
        max_intersection_dist = distances[i][j].item<float>();
      }
      most_intersect_count[i] = max_count;
      most_intersect_dist[i] = max_intersection_dist;

      auto r_most = max_count / n_intersects;
      if (r_all > 0.15) {
        if (r_most > 0.8) { // set point towards sensor
          auto dist = rt::vector_length(point_cloud[i]);
          point_cloud[i][0] *= max_intersection_dist / dist;
          point_cloud[i][1] *= max_intersection_dist / dist;
          point_cloud[i][2] *= max_intersection_dist / dist;
          point_cloud[i][3] *= 0.005;
        } else { // delete point (filtered out later)
          point_cloud[i][0] = 0;
          point_cloud[i][1] = 0;
          point_cloud[i][2] = 0;
          point_cloud[i][3] = 0;
        }
      } else { // modify intensity of unaltered point
        point_cloud[i][3] *= intensity_factor;
      }
    }
  }
}

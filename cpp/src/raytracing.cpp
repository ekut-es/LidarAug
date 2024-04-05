
#include "../include/raytracing.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>

using namespace torch_utils;
using Slice = torch::indexing::Slice;

#define NF_SPLIT_FACTOR 32

[[nodiscard]] torch::Tensor
rt::trace(torch::Tensor point_cloud, torch::Tensor noise_filter,
          torch::Tensor split_index,
          std::optional<float> intensity_factor /*= 0.9*/) {

  const auto num_points = point_cloud.size(0);
  constexpr auto num_rays = 11l;

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

[[nodiscard]] float rt::trace(const torch::Tensor &noise_filter,
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

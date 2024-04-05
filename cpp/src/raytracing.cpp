
#include "../include/raytracing.hpp"
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>

using namespace torch_utils;
using Slice = torch::indexing::Slice;

[[nodiscard]] torch::Tensor
trace(torch::Tensor point_cloud, torch::Tensor noise_filter,
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
  intersects(point_cloud, noise_filter, split_index, intersections, distances,
             distance_count, most_intersect_count, most_intersect_dist,
             num_points, intensity_factor.value_or(0.9));

  // select all points where x & y & z != 0
  const auto indices =
      (point_cloud.index({Slice(), Slice(0, 3)}) != 0).sum(1).nonzero();

  auto result = point_cloud.index({indices}).squeeze(1);

  return result;
}


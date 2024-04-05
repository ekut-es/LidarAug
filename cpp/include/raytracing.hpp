
#include "tensor.hpp"
#include <optional>
#include <torch/serialize/tensor.h>

#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

[[nodiscard]] torch::Tensor trace(torch::Tensor point_cloud,
                                  torch::Tensor noise_filter,
                                  torch::Tensor split_index,
                                  std::optional<float> intensity_factor = 0.9);

void intersects(torch::Tensor point_cloud, torch::Tensor noise_filter,
                torch::Tensor split_index, torch::Tensor intersections,
                torch::Tensor distances, torch::Tensor distance_count,
                torch::Tensor most_intersect_count,
                torch::Tensor most_intersect_dist, tensor_size_t num_points,
                float intensity_factor);

#endif // !RAYTRACING_HPP

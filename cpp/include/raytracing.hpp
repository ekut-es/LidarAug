
#include "tensor.hpp"
#include <optional>
#include <torch/serialize/tensor.h>

#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

[[nodiscard]] torch::Tensor trace(torch::Tensor point_cloud,
                                  torch::Tensor noise_filter,
                                  torch::Tensor split_index,
                                  std::optional<float> intensity_factor = 0.9);

#endif // !RAYTRACING_HPP

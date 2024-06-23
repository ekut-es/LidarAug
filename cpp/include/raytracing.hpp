
#include <torch/serialize/tensor.h>

#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

[[nodiscard]] torch::Tensor trace(const torch::Tensor &point_cloud,
                                  const torch::Tensor &noise_filter,
                                  const torch::Tensor &sort_index,
                                  float intensity_factor = 0.9);

#endif // !RAYTRACING_HPP

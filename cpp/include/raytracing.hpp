
#include "tensor.hpp"
#include <optional>
#include <torch/serialize/tensor.h>

#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

namespace rt {

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

[[nodiscard]] inline torch::Tensor mul(const torch::Tensor &v, float c) {
  // NOTE(tom): This is almost the same as `scale_points`
  return v * c;
}

[[nodiscard]] inline torch::Tensor
add(const torch::Tensor &v, const torch::Tensor &k, const torch::Tensor &l) {
  auto v_ptr = v.data_ptr<float>();
  auto k_ptr = k.data_ptr<float>();
  auto l_ptr = l.data_ptr<float>();

  return torch::tensor({v_ptr[0] + k_ptr[0] + l_ptr[0],
                        v_ptr[1] + k_ptr[1] + l_ptr[1],
                        v_ptr[2] + k_ptr[2] + l_ptr[2]});
}

[[nodiscard]] inline float scalar(const torch::Tensor &v,
                                  const torch::Tensor &k) {
  auto v_ptr = v.data_ptr<float>();
  auto k_ptr = k.data_ptr<float>();

  return (v_ptr[0] * k_ptr[0]) + (v_ptr[1] * k_ptr[1]) + (v_ptr[2] * k_ptr[2]);
}

[[nodiscard]] inline float vector_length(const torch::Tensor &v) {
  auto v_ptr = v.data_ptr<float>();

  return sqrt((v_ptr[0] * v_ptr[0]) + (v_ptr[1] * v_ptr[1]) +
              (v_ptr[2] * v_ptr[2]));
}

[[nodiscard]] inline torch::Tensor normalize(const torch::Tensor &v) {
  auto length = vector_length(v);
  auto v_ptr = v.data_ptr<float>();

  // TODO(tom): maybe do this in place? or use different datastructure?
  return torch::tensor(
      {v_ptr[0] / length, v_ptr[1] / length, v_ptr[2] / length});
}

[[nodiscard]] inline torch::Tensor cross(const torch::Tensor &v,
                                         const torch::Tensor &k) {
  auto v_ptr = v.data_ptr<float>();
  auto k_ptr = k.data_ptr<float>();

  return torch::tensor({(v_ptr[1] * k_ptr[2]) - (v_ptr[2] * k_ptr[1]),
                        (v_ptr[2] * k_ptr[0]) - (v_ptr[0] * k_ptr[2]),
                        (v_ptr[0] * k_ptr[1]) - (v_ptr[1] * k_ptr[0])});
}

[[nodiscard]] inline torch::Tensor rotate(const torch::Tensor &v,
                                          const torch::Tensor &k, float angle) {
  return add(mul(v, cos(angle)), mul(rt::cross(v, k), sin(angle)),
             mul(k, scalar(k, v) * (1 - cos(angle))));
}

[[nodiscard]] float trace(const torch::Tensor &noise_filter,
                          const torch::Tensor &beam,
                          const torch::Tensor &split_index);
} // namespace rt

#endif // !RAYTRACING_HPP

#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

#include "tensor.hpp"
#include <cstdint>
#include <torch/serialize/tensor.h>
#include <utility>

typedef enum {
  EXPONENTIAL,
  LOG_NORMAL,
  GM,
} distribution;

namespace rt {

[[nodiscard]] torch::Tensor trace(torch::Tensor point_cloud,
                                  const torch::Tensor &noise_filter,
                                  const torch::Tensor &split_index,
                                  float intensity_factor = 0.9);

void intersects(torch::Tensor point_cloud, const torch::Tensor &oise_filter,
                const torch::Tensor &split_index, torch::Tensor intersections,
                const torch::Tensor &distances, torch::Tensor distance_count,
                torch::Tensor most_intersect_count,
                torch::Tensor most_intersect_dist, tensor_size_t num_points,
                float intensity_factor);

[[nodiscard]] inline torch::Tensor mul(const torch::Tensor &v, float c) {
  // NOTE(tom): This is almost the same as `scale_points`
  return v * c;
}

[[nodiscard]] inline torch::Tensor
add(const torch::Tensor &v, const torch::Tensor &k, const torch::Tensor &l) {
  const auto *const v_ptr = v.data_ptr<float>();
  const auto *const k_ptr = k.data_ptr<float>();
  const auto *const l_ptr = l.data_ptr<float>();

  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return torch::tensor({v_ptr[0] + k_ptr[0] + l_ptr[0],
                        v_ptr[1] + k_ptr[1] + l_ptr[1],
                        v_ptr[2] + k_ptr[2] + l_ptr[2]});
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

[[nodiscard]] inline float scalar(const torch::Tensor &v,
                                  const torch::Tensor &k) {
  const auto *const v_ptr = v.data_ptr<float>();
  const auto *const k_ptr = k.data_ptr<float>();

  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return (v_ptr[0] * k_ptr[0]) + (v_ptr[1] * k_ptr[1]) + (v_ptr[2] * k_ptr[2]);
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

[[nodiscard]] inline float vector_length(const torch::Tensor &v) {
  const auto *const v_ptr = v.data_ptr<float>();

  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return sqrt((v_ptr[0] * v_ptr[0]) + (v_ptr[1] * v_ptr[1]) +
              (v_ptr[2] * v_ptr[2]));
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

[[nodiscard]] inline torch::Tensor normalize(const torch::Tensor &v) {
  const auto length = vector_length(v);
  const auto *const v_ptr = v.data_ptr<float>();

  // TODO(tom): maybe do this in place? or use different datastructure?
  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return torch::tensor(
      {v_ptr[0] / length, v_ptr[1] / length, v_ptr[2] / length});
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

[[nodiscard]] inline torch::Tensor cross(const torch::Tensor &v,
                                         const torch::Tensor &k) {
  const auto *const v_ptr = v.data_ptr<float>();
  const auto *const k_ptr = k.data_ptr<float>();

  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return torch::tensor({(v_ptr[1] * k_ptr[2]) - (v_ptr[2] * k_ptr[1]),
                        (v_ptr[2] * k_ptr[0]) - (v_ptr[0] * k_ptr[2]),
                        (v_ptr[0] * k_ptr[1]) - (v_ptr[1] * k_ptr[0])});
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

[[nodiscard]] inline torch::Tensor
rotate(const torch::Tensor &v, const torch::Tensor &k, const float angle) {
  return add(mul(v, cos(angle)), mul(rt::cross(v, k), sin(angle)),
             mul(k, scalar(k, v) * (1 - cos(angle))));
}

[[nodiscard]] float trace_beam(const torch::Tensor &noise_filter,
                               const torch::Tensor &beam,
                               const torch::Tensor &split_index);

[[nodiscard]] torch::Tensor sample_particles(int64_t num_particles,
                                             float precipitation,
                                             distribution d = EXPONENTIAL);

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
generate_noise_filter(const std::array<float, 6> &dim, uint32_t drops_per_m3,
                      float precipitation = 5.0, int32_t scale = 1,
                      distribution d = EXPONENTIAL);

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
sort_noise_filter(torch::Tensor nf);

} // namespace rt

#endif // !RAYTRACING_HPP

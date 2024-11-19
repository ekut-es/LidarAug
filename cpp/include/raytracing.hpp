#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

#include "../include/utils.hpp"
#include "tensor.hpp"
#include <array>
#include <cstdint>
#include <omp.h>
#include <torch/serialize/tensor.h>

constexpr auto nf_split_factor = 32;

enum struct distribution : std::uint8_t {
  exponential = 0,
  log_normal = 1,
  gm = 2,
};

enum struct simulation_type : std::uint8_t { snow = 0, rain = 1 };

constexpr std::array<std::pair<float, float>, 2> r_table = {{
    {0.6, 0.2},
    {0.15, 0.8},
}};

namespace rt {

[[nodiscard]] torch::Tensor trace(torch::Tensor point_cloud,
                                  const torch::Tensor &noise_filter,
                                  const torch::Tensor &split_index,
                                  simulation_type sim_t,
                                  float intensity_factor = 0.9);

void intersects(torch::Tensor point_cloud, const torch::Tensor &noise_filter,
                const torch::Tensor &split_index, torch::Tensor intersections,
                torch::Tensor distances, torch::Tensor distance_count,
                torch::Tensor most_intersect_count,
                torch::Tensor most_intersect_dist, tensor_size_t num_points,
                simulation_type sim_t, float intensity_factor);

template <typename T>
[[nodiscard]] inline auto mul(const vec3<T> &v, const float c) {
  return v * c;
}

template <typename T>
[[nodiscard]] inline auto add(const vec3<T> &v, const vec3<T> &k,
                              const vec3<T> &l) {
  return v + k + l;
}

template <typename T>
[[nodiscard]] inline float scalar(const vec3<T> &v, const vec3<T> &k) {
  return v.x * k.x + v.y * k.y + v.z * k.z;
}

[[nodiscard]] inline float scalar(const torch::Tensor &v,
                                  const torch::Tensor &k) {
  const auto *const v_ptr = v.data_ptr<float>();
  const auto *const k_ptr = k.data_ptr<float>();

  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return (v_ptr[0] * k_ptr[0]) + (v_ptr[1] * k_ptr[1]) + (v_ptr[2] * k_ptr[2]);
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

template <typename T>
[[nodiscard]] inline float vector_length(const vec3<T> &v) {
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

[[nodiscard]] inline float vector_length(const torch::Tensor &v) {
  const auto *const v_ptr = v.data_ptr<float>();

  // NOLINTBEGIN (*-pro-bounds-pointer-arithmetic)
  return sqrt((v_ptr[0] * v_ptr[0]) + (v_ptr[1] * v_ptr[1]) +
              (v_ptr[2] * v_ptr[2]));
  // NOLINTEND (*-pro-bounds-pointer-arithmetic)
}

template <typename T> [[nodiscard]] inline auto normalize(const vec3<T> &v) {
  const auto length = vector_length(v);
  return v / length;
}

template <typename T> inline auto normalize_in_place(vec3<T> &v) {
  const auto length = vector_length(v);
  return v /= length;
}

template <typename T>
[[nodiscard]] inline auto cross(const vec3<T> &v, const vec3<T> &k) {

  return vec3<T>(v.y * k.z - v.z * k.y, v.z * k.x - v.x * k.z,
                 v.x * k.y - v.y * k.x);
}

template <typename T>
[[nodiscard]] inline auto rotate(const vec3<T> &v, const vec3<T> &k,
                                 const float angle) {
  return add(mul(v, cos(angle)), mul(cross(v, k), sin(angle)),
             mul(k, scalar(k, v) * (1 - cos(angle))));
}

[[nodiscard]] float trace_beam(const torch::Tensor &noise_filter,
                               const vec3<float> &beam,
                               const torch::Tensor &split_index);

[[nodiscard]] torch::Tensor
sample_particles(int64_t num_particles, float precipitation,
                 distribution d = distribution::exponential);

template <typename DataType, c10::ScalarType TensorType>
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
sort_noise_filter(torch::Tensor nf) {

  auto split_index = torch::zeros(360 * nf_split_factor + 1, TensorType);

  nf = nf.index({nf.index({torch::indexing::Slice(), 3}).argsort()});
  nf = nf.index({nf.index({torch::indexing::Slice(), 5}).argsort(true)});

  if (!nf.is_contiguous()) {
    std::printf(
        "for performance reasons, please make sure that 'nf' is contiguous!");
    nf = torch::clone(nf, torch::MemoryFormat::Contiguous);
  }

  const auto *const nf_ptr = nf.const_data_ptr<DataType>();
  const auto row_size = nf.size(1);

#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp parallel for
      for (tensor_size_t i = 0; i < nf.size(0) - 1; i++) {

        const auto idx = i * row_size + 5;

        const auto val_at_i = nf_ptr[idx];
        const auto val_at_i_plus_one = nf_ptr[idx + row_size];

        if (val_at_i != val_at_i_plus_one) {
          split_index.index_put_(
              {static_cast<tensor_size_t>(val_at_i_plus_one)}, i + 1);
        }
      }
    }
  }

  split_index.index_put_({split_index.size(0) - 1}, nf.size(0) - 1);

  return std::make_pair(nf, split_index);
}

template <typename DataType, c10::ScalarType TensorType>
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
generate_noise_filter(const cpp_utils::distribution_ranges<float> &dim,
                      const uint32_t drops_per_m3,
                      const float precipitation = 5.0, const int32_t scale = 1,
                      const distribution d = distribution::exponential) {

  const auto total_drops =
      static_cast<int>(std::abs(dim.x_range.min - dim.x_range.max) *
                       std::abs(dim.y_range.min - dim.y_range.max) *
                       std::abs(dim.z_range.min - dim.z_range.max) *
                       static_cast<float>(drops_per_m3));

  const auto x = torch::empty({total_drops}, TensorType)
                     .uniform_(dim.x_range.min, dim.x_range.max);
  const auto y = torch::empty({total_drops}, TensorType)
                     .uniform_(dim.y_range.min, dim.y_range.max);
  const auto z = torch::empty({total_drops}, TensorType)
                     .uniform_(dim.z_range.min, dim.z_range.max);

  const auto dist =
      torch::sqrt(torch::pow(x, 2) + torch::pow(y, 2) + torch::pow(z, 2));
  const auto size = sample_particles(total_drops, precipitation, d) * scale;

  const auto index =
      (((torch::arctan2(y, x) * 180 / math_utils::PI_RAD) + 360) *
       nf_split_factor)
          .toType(torch_utils::I32) %
      (360 * nf_split_factor);
  const auto nf = torch::stack({x, y, z, dist, size, index}, -1);

  return sort_noise_filter<DataType, TensorType>(nf);
}

} // namespace rt

#endif // !RAYTRACING_HPP

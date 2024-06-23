
#ifndef WEATHER_HPP
#define WEATHER_HPP

#include "../include/raytracing.hpp"
#include <ATen/core/List.h>
#include <array>
#include <cstdint>
#include <optional>
#include <torch/serialize/tensor.h>
#include <vector>

typedef enum { DIST, CHAMFER } fog_parameter;

[[nodiscard]] std::optional<std::vector<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_parameter metric,
    float sigma, int mean);

[[nodiscard]] torch::Tensor fog(torch::Tensor point_cloud, fog_parameter metric,
                                float viewing_dist, float max_intensity = 1);

[[nodiscard]] torch::Tensor rain(torch::Tensor point_cloud,
                                 std::array<float, 6> dims, uint32_t num_drops,
                                 float precipitation, distribution d);

[[nodiscard]] torch::Tensor snow(torch::Tensor point_cloud,
                                 std::array<float, 6> dims, uint32_t num_drops,
                                 float precipitation, int32_t scale,
                                 float max_intensity);
void universal_weather(torch::Tensor point_cloud, float prob, float sigma,
                       int mean, float ext_a, float ext_b, float beta_a,
                       float beta_b, float del_a, float del_b, int int_a,
                       int int_b, int mean_int, int int_range);

#endif // !WEATHER_HPP

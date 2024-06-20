
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

#endif // !WEATHER_HPP

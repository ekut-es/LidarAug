
#ifndef WEATHER_HPP
#define WEATHER_HPP

#include "../include/raytracing.hpp"
#include <ATen/core/List.h>
#include <array>
#include <cstdint>
#include <optional>
#include <torch/serialize/tensor.h>

typedef enum { DIST, CHAMFER } fog_parameter;

[[nodiscard]] std::optional<torch::List<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_parameter metric,
    float sigma, int mean);

[[nodiscard]] torch::Tensor rain(torch::Tensor point_cloud,
                                 std::array<float, 6> dims, uint32_t num_drops,
                                 float precipitation, distribution d);

#endif // !WEATHER_HPP

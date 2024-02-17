
#ifndef WEATHER_HPP
#define WEATHER_HPP

#include <ATen/core/List.h>
#include <optional>
#include <torch/serialize/tensor.h>

typedef enum { DIST, CHAMFER } fog_metric;

[[nodiscard]] std::optional<torch::List<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_metric metric,
    float sigma, int mean);

#endif // !WEATHER_HPP

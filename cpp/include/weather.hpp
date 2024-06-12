
#ifndef WEATHER_HPP
#define WEATHER_HPP

#include <optional>
#include <torch/serialize/tensor.h>
#include <vector>

typedef enum { DIST, CHAMFER } fog_parameter;

[[nodiscard]] std::optional<std::vector<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_parameter metric,
    float sigma, int mean);

#endif // !WEATHER_HPP

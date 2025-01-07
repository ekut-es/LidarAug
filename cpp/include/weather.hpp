
#ifndef WEATHER_HPP
#define WEATHER_HPP

#include "../include/point_cloud.hpp"
#include "../include/raytracing.hpp"
#include "../include/stats.hpp"
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

typedef enum : std::uint8_t { DIST, CHAMFER } fog_parameter;

[[nodiscard]] inline auto get_intensity(const simulation_type sim_t) {

  auto rng = get_rng();

  switch (sim_t) {
  case simulation_type::snow: {
    SnowIntensityDistribution<double> d(0.11,
                                        point_cloud_data::max_intensity::get());
    return d(rng);
  }
  case simulation_type::rain: {
    std::uniform_real_distribution<double> d(0, 0.005);
    return point_cloud_data::max_intensity::get() * d(rng);
  }
  }
}

[[nodiscard]] std::optional<std::vector<torch::Tensor>>
fog(const torch::Tensor &point_cloud, float prob, fog_parameter metric,
    float sigma, int mean);

[[nodiscard]] torch::Tensor
fog(torch::Tensor point_cloud, fog_parameter metric, float viewing_dist,
    point_cloud_data::intensity_range max_intensity =
        point_cloud_data::intensity_range::MAX_INTENSITY_1);

[[nodiscard]] torch::Tensor
rain(torch::Tensor point_cloud,
     const cpp_utils::distribution_ranges<float> &dims, uint32_t num_drops,
     float precipitation, distribution d,
     point_cloud_data::intensity_range max_intensity =
         point_cloud_data::intensity_range::MAX_INTENSITY_1);

[[nodiscard]] torch::Tensor
snow(torch::Tensor point_cloud,
     const cpp_utils::distribution_ranges<float> &dims, uint32_t num_drops,
     float precipitation, int32_t scale,
     point_cloud_data::intensity_range max_intensity =
         point_cloud_data::intensity_range::MAX_INTENSITY_1);

[[nodiscard]] std::optional<torch::Tensor>
rain(torch::Tensor point_cloud, std::string_view noise_filter_path,
     uint32_t num_drops_sigma, float precipitation_sigma, float prob);

[[nodiscard]] std::optional<torch::Tensor>
snow(torch::Tensor point_cloud, std::string_view noise_filter_path,
     uint32_t num_drops_sigma, float precipitation_sigma, int32_t scale,
     float prob);

void universal_weather(torch::Tensor point_cloud, float prob, float sigma,
                       int mean, float ext_a, float ext_b, float beta_a,
                       float beta_b, float del_a, float del_b, int int_a,
                       int int_b, int mean_int, int int_range);

#endif // !WEATHER_HPP

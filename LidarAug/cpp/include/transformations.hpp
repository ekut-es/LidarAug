
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include "../include/linalg.h"
#include "../include/tensor.hpp"
#include <array>
#include <boost/math/distributions/normal.hpp>
#include <optional>
#include <random>
#include <torch/serialize/tensor.h>
#include <variant>

#define PI_DEG 180.0

typedef struct {
  tensor_size_t batch_size, num_items, num_features;
} dimensions;
typedef struct {
  float scale;
  linalg::aliases::float3 translate, rotate;
} transformations;

template <typename T> struct range {
  T min, max;
};

template <typename T> struct distribution_ranges {
  range<T> x_range, y_range, z_range, uniform_range;
};

typedef enum { UNIFORM, SALT_PEPPER, MIN, MAX } noise;

void translate(at::Tensor points, at::Tensor translation);
void scale_points(at::Tensor points, float factor);
void scale_labels(at::Tensor labels, float factor);

void translate_random(at::Tensor points, at::Tensor labels, float sigma);
void scale_random(at::Tensor points, at::Tensor labels, float sigma,
                  float max_scale);
void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob);

/**
 * Introduces random noise to a point cloud.
 *
 * @param points is a (n, 4) tensor representing the point cloud
 * @param sigma  TODO
 * @param ranges TODO
 * @param type   The type of noise that is to be introduced
 */
void random_noise(at::Tensor points, float sigma,
                  const distribution_ranges<float> &ranges, noise type);
void thin_out(at::Tensor points, float sigma);
void rotate_random(at::Tensor points, at::Tensor labels, float sigma);

/**
 * Returns a random number generator using `std::random_device` as a seed and
 * `std::mt18837` as a generator.
 *
 * @returns an instance of a `std::mt19937` instance
 */
[[nodiscard]] inline std::mt19937 get_rng() noexcept {
  std::random_device seed;
  std::mt19937 rng(seed());

  return rng;
}

/**
 * Function to draw a number of values from a provided distribution.
 *
 * @param dist              A reference to one of the following distributions:
 *                            - uniform_int_distribution
 *                            - uniform_real_distribution
 *                            - normal_distribution
 * @param number_of_values  Optional argument to draw more than one value.
 * @param force             Forces the function to use a vector even if there is
 *                          only one value.
 *
 * @returns A value of type T or multiple values of type T wrapped in a vector.
 */
template <typename T, typename D>
[[nodiscard]] static inline std::variant<std::vector<T>, T>
draw_values(D &dist, std::optional<std::size_t> number_of_values = 1,
            std::optional<bool> force = false) noexcept {

  static_assert(std::is_base_of<std::uniform_int_distribution<T>, D>::value ||
                std::is_base_of<std::uniform_real_distribution<T>, D>::value ||
                std::is_base_of<std::normal_distribution<T>, D>::value ||
                "'dist' does not satisfy the type constaints!");

  auto rng = get_rng();

  std::size_t n = number_of_values.value_or(1);

  if (n > 1 || force.value_or(false)) {
    std::vector<T> numbers;
    numbers.reserve(n);

    for (std::size_t i = 0; i < n; i++) {
      auto v = dist(rng);
      numbers.emplace_back(v);
    }

    return numbers;

  } else {
    return dist(rng);
  }
}

[[nodiscard]] inline float get_truncated_normal_value(
    std::optional<float> mean = 0, std::optional<float> sd = 1,
    std::optional<float> low = 0, std::optional<float> up = 10) {

  auto rng = get_rng();

  // create normal distribution
  boost::math::normal_distribution<float> nd(mean.value(), sd.value());

  // get upper and lower bounds using the cdf, which are the probabilities for
  // the values being within those bounds
  auto lower_cdf = boost::math::cdf(nd, low.value());
  auto upper_cdf = boost::math::cdf(nd, up.value());

  // create uniform distribution based on those bounds, plotting the
  // probabilities
  std::uniform_real_distribution<double> ud(lower_cdf, upper_cdf);

  // sample uniform distribution, returning a uniformly distributed value
  // between upper and lower
  auto ud_sample = ud(rng);

  // use the quantile function (inverse of cdf, so equal to ppf) to 'convert'
  // the sampled probability into its corresponding value
  auto sample = boost::math::quantile(nd, ud_sample);

  return sample;
}

/**
 * Generates a rotation matrix around the 'z' axis (yaw) from the provided
 * angle.
 *
 * @param angle is the angle (in radians)
 *
 * @returns a 3x3 rotation matrix (in form of a torch::Tensor)
 */
[[nodiscard]] inline torch::Tensor rotate_yaw(float angle) {

  float cos_angle = cos(angle);
  float sin_angle = sin(angle);

  auto rotation = torch::tensor({{cos_angle, 0.0f, sin_angle},
                                 {0.0f, 1.0f, 0.0f},
                                 {-sin_angle, 0.0f, cos_angle}});
  return rotation;
}

[[nodiscard]] inline float to_rad(float angle) noexcept {
  return angle * (M_PI / PI_DEG);
}

#endif // !TRANSFORMATIONS_HPP

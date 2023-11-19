
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include "../include/linalg.h"
#include <array>
#include <boost/math/distributions/normal.hpp>
#include <cstdint>
#include <optional>
#include <random>
#include <torch/serialize/tensor.h>
#include <variant>

typedef struct {
  int batch_size, num_points, num_point_features;
} dimensions;
typedef struct {
  float scale;
  linalg::aliases::float3 translate, rotate;
} transformations;

typedef enum { UNIFORM, SALT_PEPPER, MIN, MAX } noise;

void translate(at::Tensor points, at::Tensor translation);
void scale_points(at::Tensor points, double factor);
void scale_labels(at::Tensor labels, double factor);

void translate_random(at::Tensor points, at::Tensor labels, double sigma);
void scale_random(at::Tensor points, at::Tensor labels, double sigma,
                  double max_scale);
void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob);

void random_noise(at::Tensor points, double sigma,
                  const std::array<double, 8> &ranges, noise type);
void rotate_random(at::Tensor points, at::Tensor labels, double sigma);

inline std::mt19937 get_rng() {
  // seed
  std::random_device d;

  // random number generator
  std::mt19937 gen(d());

  return gen;
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
static inline std::variant<std::vector<T>, T>
draw_values(D &dist, std::optional<std::size_t> number_of_values = 1,
            std::optional<bool> force = false) {

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

inline double get_normal(double scale, double mean) {
  auto rng = get_rng();

  std::normal_distribution<double> dist(mean, scale);

  return dist(rng);
}

inline double get_truncated_normal_value(std::optional<double> mean = 0,
                                         std::optional<double> sd = 1,
                                         std::optional<double> low = 0,
                                         std::optional<double> up = 10) {

  auto rng = get_rng();

  // create normal distribution
  boost::math::normal_distribution<double> nd(mean.value(), sd.value());

  // get upper and lower bounds using the cdf, which are the probabilities for
  // the values being within those bounds
  double lower_cdf = boost::math::cdf(nd, low.value());
  double upper_cdf = boost::math::cdf(nd, up.value());

  // create uniform distribution based on those bounds, plotting the
  // probabilities
  std::uniform_real_distribution<double> ud(lower_cdf, upper_cdf);

  // sample uniform distribution, returning a uniformly distributed value
  // between upper and lower
  double ud_sample = ud(rng);

  // use the quantile function (inverse of cdf, so equal to ppf) to 'convert'
  // the sampled probability into its corresponding value
  double sample = boost::math::quantile(nd, ud_sample);

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
inline torch::Tensor rotate_yaw(double angle) {

  auto cos_angle = cos(angle);
  auto sin_angle = sin(angle);

  auto rotation = torch::tensor({{cos_angle, 0.0, sin_angle},
                                 {0.0, 1.0, 0.0},
                                 {-sin_angle, 0.0, cos_angle}});
  return rotation;
}

inline double to_rad(double angle) { return angle * (M_PI / 180.0); }

#endif // !TRANSFORMATIONS_HPP

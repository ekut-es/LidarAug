
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include <boost/math/distributions/normal.hpp>
#include <cstdint>
#include <optional>
#include <random>
#include <torch/serialize/tensor.h>

typedef struct {
  float x, y, z;
} vec;
typedef struct {
  int batch_size, num_points, num_point_features;
} dimensions;
typedef struct {
  float scale;
  vec translate, rotate;
} transformations;

typedef enum { UNIFORM, SALT_PEPPER, MIN, MAX } noise;

void translate(at::Tensor points, at::Tensor translation);
void scale_points(at::Tensor points, double factor);
void scale_labels(at::Tensor labels, double factor);

void translate_random(at::Tensor points, at::Tensor labels, double scale);
void scale_random(at::Tensor points, at::Tensor labels, double sigma,
                  double max_scale);
void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob);

void random_noise(at::Tensor points, double sigma,
                  const std::array<double, 8> &ranges, noise type);

inline std::mt19937 get_rng() {
  // seed
  std::random_device d;

  // random number generator
  std::mt19937 gen(d());

  return gen;
}

template <typename T>
static inline std::vector<T>
draw_uniform_values(const std::uniform_real_distribution<T> &dist,
                    std::size_t number_of_values) {
  auto rng = get_rng();

  std::vector<T> numbers;
  numbers.reserve(number_of_values);

  for (std::size_t i = 0; i < number_of_values; i++) {
    numbers.emplace_back(dist(rng));
  }

  numbers;
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

#endif // !TRANSFORMATIONS_HPP

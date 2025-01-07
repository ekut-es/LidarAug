/**
 * Header file for statistics.
 */

#ifndef STATS_HPP
#define STATS_HPP

#include "tensor.hpp"
#include "utils.hpp"
#include <ATen/ops/empty.h>
#include <algorithm>
#include <boost/math/distributions/normal.hpp>
#include <c10/core/ScalarType.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include <torch/torch.h>
#include <variant>
#include <vector>

namespace constants {
constexpr auto seed = 123u;
} // namespace constants

#define HUNDRED_PERCENT 100

// constants for `std::get` for `draw_values`
#define VECTOR 0
#define VALUE 1

template <typename T = double, bool Interpolate = true>
class SnowIntensityDistribution {
public:
  SnowIntensityDistribution(float min, float max, size_t resolution = 256)
      : min_(min), max_(max), _d(0, 1) {
    if (min >= max)
      throw std::runtime_error(
          "Invalid bounds (hint: min must be smaller than max)");

    _sampled_cdf.resize(resolution + 1);

    const T cdf_low = cdf(min);
    const T cdf_high = cdf(max);

    size_t i = 0;

    std::generate(_sampled_cdf.begin(), _sampled_cdf.end(),
                  [&i, resolution, max, min, this, cdf_low, cdf_high]() {
                    const T x =
                        static_cast<T>(i) / resolution * (max - min) + min;
                    const T p = (cdf(x) - cdf_low) / (cdf_high - cdf_low);
                    i++;
                    return Sample{p, x};
                  });
  }
  SnowIntensityDistribution(SnowIntensityDistribution &&) = default;
  SnowIntensityDistribution(const SnowIntensityDistribution &) = default;
  SnowIntensityDistribution &operator=(SnowIntensityDistribution &&) = default;
  SnowIntensityDistribution &
  operator=(const SnowIntensityDistribution &) = default;
  template <typename Generator> T operator()(Generator &g) {

    T cdf_val = _d(g);
    auto s =
        std::upper_bound(_sampled_cdf.begin(), _sampled_cdf.end(), cdf_val);
    auto bs = s - 1;
    if (Interpolate && bs >= _sampled_cdf.begin()) {
      const T r = (cdf_val - bs->prob) / (s->prob - bs->prob);
      return r * bs->value + (1 - r) * s->value;
    }
    return s->value;
  }

  [[nodiscard]] T cdf(const T x) {
    return 0.5 * erf(1.08953 * log(4.90196 * (x - 0.105)));
  }

private:
  const float min_, max_;

  struct Sample {
    T prob, value;
    friend bool operator<(T p, const Sample &s) { return p < s.prob; }
  };

  std::vector<Sample> _sampled_cdf;
  std::uniform_real_distribution<> _d;
};

/**
 * Returns a random number generator using `std::random_device` as a seed and
 * `std::mt18837` as a generator.
 *
 * @returns an instance of a `std::mt19937` instance
 */
[[nodiscard]] inline std::mt19937 get_rng() noexcept {

#ifdef TEST_RNG
  std::mt19937 rng(constants::seed);
#else
  std::random_device seed;
  std::mt19937 rng(seed());
#endif

  return rng;
}

/**
 * Function to draw a number of values from a provided distribution.
 *
 * @param dist              A reference to one of the following distributions:
 *                            - uniform_int_distribution
 *                            - uniform_real_distribution
 *                            - normal_distribution
 *                            - exponential_distribution
 * @param number_of_values  Optional argument to draw more than one value.
 * @param force             Forces the function to use a vector even if there is
 *                          only one value.
 *
 * @returns A value of type T or multiple values of type T wrapped in a vector.
 */
template <typename T, typename D>
[[nodiscard]] static inline std::variant<std::vector<T>, T>
draw_values(D &dist, std::size_t number_of_values = 1,
            bool force = false) noexcept {

  static_assert(std::is_base_of_v<std::uniform_int_distribution<T>, D> ||
                std::is_base_of_v<std::uniform_real_distribution<T>, D> ||
                std::is_base_of_v<std::normal_distribution<T>, D> ||
                std::is_base_of_v<std::exponential_distribution<T>, D> ||
                "'dist' does not satisfy the type constaints!");

  auto rng = get_rng();

  std::size_t n = number_of_values;

  if (n > 1) {
    std::vector<T> numbers(n);

    auto draw = [&dist, &rng]() { return dist(rng); };
    std::generate(numbers.begin(), numbers.end(), draw);

    return numbers;
  } else if (force) {
    return std::vector<T>{dist(rng)};
  } else {
    return dist(rng);
  }
}

/**
 * Function to draw a number of values from a provided distribution.
 *
 * @param dist              A reference to one of the following distributions:
 *                            - uniform_int_distribution
 *                            - uniform_real_distribution
 *                            - normal_distribution
 *                            - exponential_distribution
 * @param number_of_values  Optional argument to draw more than one value.
 *
 * @returns A `torch::Tensor` containing all the drawn values.
 */
template <typename T, c10::ScalarType type, typename D>
[[nodiscard]] static inline torch::Tensor
draw_values(D &dist, tensor_size_t number_of_values = 1) {

  static_assert(std::is_base_of_v<std::uniform_int_distribution<T>, D> ||
                std::is_base_of_v<std::uniform_real_distribution<T>, D> ||
                std::is_base_of_v<std::normal_distribution<T>, D> ||
                std::is_base_of_v<std::exponential_distribution<T>, D> ||
                "'dist' does not satisfy the type constaints!");

  auto rng = get_rng();

  auto result = torch::empty({number_of_values}, type);
  auto data = result.data_ptr<T>();

  for (tensor_size_t i = 0; i < number_of_values; i++) {
    data[i] = dist(rng);
  }

  return result;
}

[[nodiscard]] inline float get_truncated_normal_value(float mean = 0,
                                                      float sd = 1,
                                                      float low = 0,
                                                      float up = 10) {

  auto rng = get_rng();

  // create normal distribution
  boost::math::normal_distribution<float> nd(mean, sd);

  // get upper and lower bounds using the cdf, which are the probabilities for
  // the values being within those bounds
  auto lower_cdf = boost::math::cdf(nd, low);
  auto upper_cdf = boost::math::cdf(nd, up);

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
 * Draws a fixed amount of values from a pool of potential values.
 *
 * @param size       defines the end of the range of total values: [0; size)
 * @param num_values is the number of values to be drawn
 *
 * @returns a vector containing `num_values` random unique values in [0; size)
 */
template <typename T>
[[nodiscard]] inline std::vector<T>
draw_unique_uniform_values(std::size_t size, std::size_t num_values) {
  auto rng = get_rng();

  if (size < num_values)
    throw std::invalid_argument("num_values cannot exceed size.");

  std::vector<T> values(num_values);
  std::vector<T> samples(size);
  std::iota(samples.begin(), samples.end(), 0);

  std::sample(samples.begin(), samples.end(), values.begin(), num_values, rng);

  return values;
}

[[nodiscard]] inline torch::Tensor
inverted_lognormal_cdf(const torch::Tensor &d, const float r) {
  return std::pow(r, 0.23) *
         torch::exp((math_utils::sqrt2 * std::log(1.43 - (0.0003 * r)) *
                     torch::erfinv((0.0116279 * d) / (std::pow(r, 0.22)))) -
                    0.328504);
}

[[nodiscard]] inline torch::Tensor
inverted_exponential_cdf(const torch::Tensor &d, const float r) {
  return -0.243902 * std::pow(r, 0.21) *
         torch::log(0.0005124998718750320 * d * std::pow(r, -0.21));
}

[[nodiscard]] inline torch::Tensor
inverted_exponential_gm(const torch::Tensor &d, const float r) {
  return -0.436681 * std::pow(r, 0.48) *
             torch::log(0.000916002564807181 * d * std::pow(r, 0.46)) -
         5.9143581981431375;
}

#endif // !STATS_HPP

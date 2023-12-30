/**
 * Header file for statistics.
 */

#ifndef STATS_HPP
#define STATS_HPP

#include <boost/math/distributions/normal.hpp>
#include <optional>
#include <random>
#include <variant>

/**
 * Returns a random number generator using `std::random_device` as a seed and
 * `std::mt18837` as a generator.
 *
 * @returns an instance of a `std::mt19937` instance
 */
[[nodiscard]] inline std::mt19937 get_rng() noexcept {

#ifdef TEST_RNG
  auto seed = 123u;
  std::mt19937 rng(seed);
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

  if (n > 1) {
    std::vector<T> numbers(n);

    auto draw_values = [&dist, &rng]() { return dist(rng); };
    std::generate(numbers.begin(), numbers.end(), draw_values);

    return numbers;
  } else if (force.value_or(false)) {
    return std::vector<T>{dist(rng)};
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
 * Draws a fixed amount of values from a pool of potential values.
 *
 * NOTE: this can generate a lot of values and might be inefficient for big
 * `size`s but small `num_values`!
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

  std::vector<T> values(size);
  std::generate(values.begin(), values.end(),
                [value = 0]() mutable { return value++; });
  std::shuffle(values.begin(), values.end(), rng);

  values.resize(num_values);

  return values;
}

#endif // !STATS_HPP
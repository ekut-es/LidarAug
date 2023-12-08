
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include "../include/linalg.h"
#include "../include/tensor.hpp"
#include <torch/serialize/tensor.h>

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

void translate(at::Tensor points, const at::Tensor &translation);
void scale_points(at::Tensor points, float factor);
void scale_labels(at::Tensor labels, float factor);

void translate_random(at::Tensor points, at::Tensor labels, float sigma);
void scale_random(at::Tensor points, at::Tensor labels, float sigma,
                  float max_scale);
void scale_local(at::Tensor point_cloud, at::Tensor labels, float sigma,
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
void random_noise(at::Tensor &points, float sigma,
                  const distribution_ranges<float> &ranges, noise type);

/**
 * Randomly genereates a percentage from a norma distribution, which determines
 * how many items should be 'thinned out'. From that percentage random indeces
 * are uniformly drawn (in a random order, where each index is unique).
 *
 * Finally a new tensor is created containing the items present at those
 * indeces.
 *
 * @param points is the point cloud.
 * @param sigma  is the standard diviation of the distribution that genereates
 *               the percentage.
 *
 * @returns a new tensor containing the new data
 */
[[nodiscard]] torch::Tensor thin_out(at::Tensor points, float sigma);

void rotate_random(at::Tensor points, at::Tensor labels, float sigma);

#endif // !TRANSFORMATIONS_HPP

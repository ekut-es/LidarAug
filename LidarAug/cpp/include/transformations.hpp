
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

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

void translate(at::Tensor points, at::Tensor translation);
void translate_random(at::Tensor points, at::Tensor labels, double scale);

inline double get_normal(double scale, double mean) {
  // seed
  std::random_device d;

  // random number generator
  std::mt19937 gen(d());

  std::normal_distribution<double> dist(mean, scale);

  return dist(gen);
}

#endif // !TRANSFORMATIONS_HPP

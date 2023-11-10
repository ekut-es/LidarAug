
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

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

#endif // !TRANSFORMATIONS_HPP

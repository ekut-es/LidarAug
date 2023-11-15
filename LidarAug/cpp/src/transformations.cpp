#include "../include/transformations.hpp"
#include <math.h>
#include <stdlib.h>

void translate(at::Tensor points, at::Tensor translation) {
  dimensions dims = {static_cast<int>(points.size(0)),
                     static_cast<int>(points.size(1)),
                     static_cast<int>(points.size(2))};
  float *t = translation.data_ptr<float>();
  vec translate = {t[0], t[1], t[2]};

  // translate all point clouds in a batch by the same amount
  for (int i = 0; i < dims.batch_size; i++) {
    for (int j = 0; j < dims.num_points; j++) {
      points.index({i, j, 0}) += translate.x;
      points.index({i, j, 1}) += translate.y;
      points.index({i, j, 2}) += translate.z;
    }
  }
}

void scale_points(at::Tensor points, double factor) {
  dimensions dims = {static_cast<int>(points.size(0)),
                     static_cast<int>(points.size(1)),
                     static_cast<int>(points.size(2))};

  // scale all point clouds in a batch by the same amount
  for (int i = 0; i < dims.batch_size; i++) {
    for (int j = 0; j < dims.num_points; j++) {
      points.index({i, j, 0}) *= factor;
      points.index({i, j, 1}) *= factor;
      points.index({i, j, 2}) *= factor;
    }
  }
}

void scale_labels(at::Tensor labels, double factor) {
  // scale all the labels in a batch by the same amount
  for (int i = 0; i < static_cast<int>(labels.size(0)); i++) {
    labels.index({i, 0}) *= factor;
    labels.index({i, 1}) *= factor;
    labels.index({i, 2}) *= factor;
    labels.index({i, 3}) *= factor;
    labels.index({i, 4}) *= factor;
    labels.index({i, 5}) *= factor;
  }
}

void translate_random(at::Tensor points, at::Tensor labels, double sigma) {

  double x_translation = get_normal(sigma, 0);
  double y_translation = get_normal(sigma, 0);
  double z_translation = get_normal(sigma, 0);

  auto translation = at::tensor({x_translation, y_translation, z_translation});

  translate(points, translation);
  translate(labels, translation);

  // NOTE(tom): coop boxes not implemented
}

void scale_random(at::Tensor points, at::Tensor labels, double sigma,
                  double max_scale) {

  double scale_factor =
      get_truncated_normal_value(1, sigma, (1 / max_scale), max_scale);

  scale_points(points, scale_factor);
  scale_labels(labels, scale_factor);

  // NOTE(tom): coop boxes not implemented
}

bool flip_random(at::Tensor points, at::Tensor labels, std::size_t prob) {

  if (prob < 101) [[likely]] {
    std::size_t r = rand() % 100 + 1;

    if (prob > r) {
      dimensions dims = {static_cast<int>(points.size(0)),
                         static_cast<int>(points.size(1)),
                         static_cast<int>(points.size(2))};

      for (int i = 0; i < dims.batch_size; i++) {
        for (int j = 0; j < dims.num_points; j++) {
          points.index({i, j, 1}) *= -1;
        }
      }
      for (int i = 0; i < static_cast<int>(labels.size(0)); i++) {
        labels.index({i, 1}) *= -1;
        labels.index({i, 6}) = (labels.index({i, 6}) + M_PI) % (2 * M_PI);
      }
    }

    return true;
  } else {
    return false;
  }
}

// uncomment this to include the bindings to build the python library
// #define BUILD
#ifdef BUILD
#include "../include/bindings.hpp"
#else
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

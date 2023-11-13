#include "../include/transformations.hpp"

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

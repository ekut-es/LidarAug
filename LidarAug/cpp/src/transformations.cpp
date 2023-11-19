#include "../include/transformations.hpp"
#include "../include/label.hpp"
#include "../include/point_cloud.hpp"
#include <math.h>

void translate(at::Tensor points, at::Tensor translation) {
  dimensions dims = {static_cast<int>(points.size(0)),
                     static_cast<int>(points.size(1)),
                     static_cast<int>(points.size(2))};
  float *t = translation.data_ptr<float>();
  linalg::aliases::float3 translate{t[0], t[1], t[2]};

  // translate all point clouds in a batch by the same amount
  for (int i = 0; i < dims.batch_size; i++) {
    for (int j = 0; j < dims.num_points; j++) {
      points.index({i, j, POINT_CLOUD_X_IDX}) += translate.x;
      points.index({i, j, POINT_CLOUD_Y_IDX}) += translate.y;
      points.index({i, j, POINT_CLOUD_Z_IDX}) += translate.z;
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
      points.index({i, j, POINT_CLOUD_X_IDX}) *= factor;
      points.index({i, j, POINT_CLOUD_Y_IDX}) *= factor;
      points.index({i, j, POINT_CLOUD_Z_IDX}) *= factor;
    }
  }
}

void scale_labels(at::Tensor labels, double factor) {
  // scale all the labels in a batch by the same amount
  for (int i = 0; i < static_cast<int>(labels.size(0)); i++) {
    labels.index({i, LABEL_X_IDX}) *= factor;
    labels.index({i, LABEL_Y_IDX}) *= factor;
    labels.index({i, LABEL_Z_IDX}) *= factor;
    labels.index({i, LABEL_W_IDX}) *= factor;
    labels.index({i, LABEL_H_IDX}) *= factor;
    labels.index({i, LABEL_L_IDX}) *= factor;
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

void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob) {

  auto rng = get_rng();
  std::uniform_int_distribution<std::size_t> distrib(0, 100);
  auto rand = distrib(rng);

  if (prob > rand) {
    dimensions dims = {static_cast<int>(points.size(0)),
                       static_cast<int>(points.size(1)),
                       static_cast<int>(points.size(2))};

    for (int i = 0; i < dims.batch_size; i++) {
      for (int j = 0; j < dims.num_points; j++) {
        points.index({i, j, POINT_CLOUD_Y_IDX}) *= -1;
      }
    }
    for (int i = 0; i < static_cast<int>(labels.size(0)); i++) {
      labels.index({i, LABEL_Y_IDX}) *= -1;
      labels.index({i, LABEL_ANGLE_IDX}) =
          (labels.index({i, 6}) + M_PI) % (2 * M_PI);
    }
  }
}

void random_noise(at::Tensor points, double sigma,
                  const std::array<double, 8> &ranges, noise type) {

  auto rng = get_rng();
  std::normal_distribution<double> normal(0.0, sigma);
  std::uniform_real_distribution<float> x_distrib(ranges[0], ranges[1]);
  std::uniform_real_distribution<float> y_distrib(ranges[2], ranges[3]);
  std::uniform_real_distribution<float> z_distrib(ranges[4], ranges[5]);

  const std::size_t num_points = std::abs(normal(rng));

  const auto x = std::get<0>(draw_values<float>(x_distrib, num_points, true));
  const auto y = std::get<0>(draw_values<float>(y_distrib, num_points, true));
  const auto z = std::get<0>(draw_values<float>(z_distrib, num_points, true));

  std::vector<float> noise_values;
  noise_values.reserve(num_points);

  switch (type) {
  case UNIFORM: {
    std::uniform_real_distribution<float> ud(ranges[6], ranges[7]);
    noise_values = std::get<0>(draw_values<float>(ud, num_points, true));
    break;
  }
  case SALT_PEPPER: {
    const auto salt_len = num_points / 2;
    const std::vector<float> salt(salt_len, 0);
    const std::vector<float> pepper(num_points - salt_len, 255);

    noise_values.insert(noise_values.begin(), salt.begin(), salt.end());
    noise_values.insert(noise_values.end(), pepper.begin(), pepper.end());
    break;
  }
  case MIN: {
    std::fill(noise_values.begin(), noise_values.end(), 0);
    break;
  }
  case MAX: {
    std::fill(noise_values.begin(), noise_values.end(), 255);
    break;
  }
  }

  std::vector<linalg::aliases::float4> stacked_values;
  stacked_values.reserve(num_points);

  // 'stack' x, y, z and noise (same as np.stack((x, y, z, noise_values),
  // axis=-1))
  for (std::size_t i = 0; i < num_points; i++) {
    const linalg::aliases::float4 vals{x[i], y[i], z[i], noise_values[i]};
    stacked_values.emplace_back(vals);
  }

  // TODO concat

  /*
   * concat along the rows
  self.point_cloud = np.concatenate((self.point_cloud, noise), axis=0)
  */
}

void rotate_random(at::Tensor points, at::Tensor labels, double sigma) {
  auto rot_angle = get_truncated_normal_value(0, sigma, -180, 180);
  auto angle_rad = to_rad(rot_angle);

  auto rotation = rotate_yaw(angle_rad);

  for (tensor_size_t i = 0; i < points.size(0); i++) {
    for (tensor_size_t j = 0; j < points.size(1); j++) {

      auto points_vec = points[i][j];

      points[i][j] = torch::matmul(points_vec, rotation);
    }
  }

  for (tensor_size_t i = 0; i < labels.size(0); i++) {
    auto label = labels[i];
    auto label_vec = torch::tensor({labels[i][LABEL_X_IDX].item<double>(),
                                    labels[i][LABEL_Y_IDX].item<double>(),
                                    labels[i][LABEL_X_IDX].item<double>()});
    labels[i] = torch::matmul(label_vec, rotation);

    label[LABEL_ANGLE_IDX] = (label[LABEL_ANGLE_IDX] + angle_rad) % (2 * M_PI);
  }

  // NOTE(tom): coop boxes not implemented
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

#include "../include/transformations.hpp"
#include "../include/label.hpp"
#include "../include/point_cloud.hpp"
#include "../include/utils.hpp"
#include <math.h>

void translate(at::Tensor points, at::Tensor translation) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};
  auto *t = translation.data_ptr<float>();
  linalg::aliases::float3 translate{t[0], t[1], t[2]};

  // translate all point clouds in a batch by the same amount
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      points.index({i, j, POINT_CLOUD_X_IDX}) += translate.x;
      points.index({i, j, POINT_CLOUD_Y_IDX}) += translate.y;
      points.index({i, j, POINT_CLOUD_Z_IDX}) += translate.z;
    }
  }
}

void scale_points(at::Tensor points, float factor) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};

  // scale all point clouds by the same amount
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      points.index({i, j, POINT_CLOUD_X_IDX}) *= factor;
      points.index({i, j, POINT_CLOUD_Y_IDX}) *= factor;
      points.index({i, j, POINT_CLOUD_Z_IDX}) *= factor;
    }
  }
}

void scale_labels(at::Tensor labels, float factor) {
  dimensions dims = {labels.size(0), labels.size(1), labels.size(2)};

  // scale all the labels by the same amount
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      labels.index({i, j, LABEL_X_IDX}) *= factor;
      labels.index({i, j, LABEL_Y_IDX}) *= factor;
      labels.index({i, j, LABEL_Z_IDX}) *= factor;
      labels.index({i, j, LABEL_W_IDX}) *= factor;
      labels.index({i, j, LABEL_H_IDX}) *= factor;
      labels.index({i, j, LABEL_L_IDX}) *= factor;
    }
  }
}

void translate_random(at::Tensor points, at::Tensor labels, float sigma) {

  std::normal_distribution<float> dist(sigma, 0);

  auto x_translation = std::get<1>(draw_values<float>(dist));
  auto y_translation = std::get<1>(draw_values<float>(dist));
  auto z_translation = std::get<1>(draw_values<float>(dist));

  auto translation = at::tensor({x_translation, y_translation, z_translation});

  translate(points, translation);
  translate(labels, translation);

  // NOTE(tom): coop boxes not implemented
}

void scale_random(at::Tensor points, at::Tensor labels, float sigma,
                  float max_scale) {

  auto scale_factor =
      get_truncated_normal_value(1, sigma, (1 / max_scale), max_scale);

  scale_points(points, scale_factor);
  scale_labels(labels, scale_factor);

  // NOTE(tom): coop boxes not implemented
}

void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob) {

  auto rng = get_rng();
  std::uniform_int_distribution<std::size_t> distrib(0, 99);
  auto rand = distrib(rng);

  if (prob > rand) {
    dimensions point_dims = {points.size(0), points.size(1), points.size(2)};

    for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {
      for (tensor_size_t j = 0; j < point_dims.num_items; j++) {
        points.index({i, j, POINT_CLOUD_Y_IDX}) *= -1;
      }
    }

    dimensions label_dims = {labels.size(0), labels.size(1), labels.size(2)};
    for (tensor_size_t i = 0; i < label_dims.batch_size; i++) {
      for (tensor_size_t j = 0; j < label_dims.num_items; j++) {
        labels.index({i, j, LABEL_Y_IDX}) *= -1;
        labels.index({i, j, LABEL_ANGLE_IDX}) =
            (labels.index({i, j, LABEL_ANGLE_IDX}) + M_PI) % (2 * M_PI);
      }
    }
  }
}

void random_noise(at::Tensor points, float sigma,
                  const distribution_ranges<float> &ranges, noise type) {

  auto rng = get_rng();
  std::normal_distribution<float> normal(0.0, sigma);
  std::uniform_real_distribution<float> x_distrib(ranges.x_range.min,
                                                  ranges.x_range.max);
  std::uniform_real_distribution<float> y_distrib(ranges.y_range.min,
                                                  ranges.y_range.max);
  std::uniform_real_distribution<float> z_distrib(ranges.z_range.min,
                                                  ranges.z_range.max);

  // iterate over batches
  for (tensor_size_t batch_num = 0; batch_num < points.size(0); batch_num++) {

    const std::size_t num_points = std::abs(normal(rng));

    const auto x = std::get<0>(draw_values<float>(x_distrib, num_points, true));
    const auto y = std::get<0>(draw_values<float>(y_distrib, num_points, true));
    const auto z = std::get<0>(draw_values<float>(z_distrib, num_points, true));

    std::vector<float> noise_intensity;
    noise_intensity.reserve(num_points);

    switch (type) {
    case UNIFORM: {
      std::uniform_real_distribution<float> ud(ranges.uniform_range.min,
                                               ranges.uniform_range.max);
      noise_intensity = std::get<0>(draw_values<float>(ud, num_points, true));
      break;
    }
    case SALT_PEPPER: {
      const auto salt_len = num_points / 2;
      const std::vector<float> salt(salt_len, 0);
      const std::vector<float> pepper(num_points - salt_len, 255);

      noise_intensity.insert(noise_intensity.begin(), salt.begin(), salt.end());
      noise_intensity.insert(noise_intensity.end(), pepper.begin(),
                             pepper.end());
      break;
    }
    case MIN: {
      std::fill(noise_intensity.begin(), noise_intensity.end(), 0);
      break;
    }
    case MAX: {
      std::fill(noise_intensity.begin(), noise_intensity.end(), 255);
      break;
    }
    }

    std::vector<linalg::aliases::float4> stacked_values;
    stacked_values.reserve(num_points);

    // 'stack' x, y, z and noise (same as np.stack((x, y, z, noise_intensity),
    // axis=-1))
    for (std::size_t i = 0; i < num_points; i++) {
      const linalg::aliases::float4 vals{x[i], y[i], z[i], noise_intensity[i]};
      stacked_values.emplace_back(vals);
    }

    // NOTE(tom): this involves copying all the data over. Maybe more efficent
    // to work with tensors from the beginning?
    auto noise_tensor = torch::from_blob(
        stacked_values.data(),
        {static_cast<tensor_size_t>(stacked_values.size())}, torch::kFloat32);

    // concatenate points
    torch::stack({points[batch_num], noise_tensor});
  }
}

void rotate_random(at::Tensor points, at::Tensor labels, float sigma) {

  dimensions point_dims = {points.size(0), points.size(1), points.size(2)};
  auto rot_angle = get_truncated_normal_value(0, sigma, -PI_DEG, PI_DEG);
  auto angle_rad = to_rad(rot_angle);

  auto rotation = rotate_yaw(angle_rad);

  for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < point_dims.num_items; j++) {

      auto points_vec = points[i][j];

      points[i][j] = torch::matmul(points_vec, rotation);
    }
  }

  dimensions label_dims = {labels.size(0), labels.size(1), labels.size(2)};
  for (tensor_size_t i = 0; i < label_dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < label_dims.num_items; j++) {
      auto label = labels[i][j];
      auto label_vec = torch::tensor({label[LABEL_X_IDX].item<double>(),
                                      label[LABEL_Y_IDX].item<double>(),
                                      label[LABEL_X_IDX].item<double>()});
      torch::matmul(label_vec, rotation);

      label[LABEL_ANGLE_IDX] =
          (label[LABEL_ANGLE_IDX] + angle_rad) % (2 * M_PI);
    }
  }

  // NOTE(tom): coop boxes not implemented
}

void thin_out(at::Tensor points, float sigma) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};

  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    auto percent = get_truncated_normal_value(0, sigma, 0, 1);
    std::uniform_int_distribution<tensor_size_t> ud(
        dims.num_items, dims.num_items * (1 - percent));

    auto idx = std::get<1>(draw_values<tensor_size_t>(ud));

    // remove first n-1 elements
    points[i].slice(0, idx - 1);
  }
}

#ifdef BUILD_MODULE
#include "../include/bindings.hpp"
#else
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

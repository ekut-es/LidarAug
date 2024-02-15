#include "../include/transformations.hpp"
#include "../include/label.hpp"
#include "../include/linalg.h"
#include "../include/point_cloud.hpp"
#include "../include/stats.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <cmath>
#include <torch/types.h>

using f3 = linalg::aliases::float3;

void translate(at::Tensor points, const at::Tensor &translation) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};
  const float *const translation_ptr = translation.data_ptr<float>();
  const float x_translation =
      translation_ptr[0]; // NOLINT: Allow pointer arithmetic to access contents
  const float y_translation =
      translation_ptr[1]; // NOLINT: Allow pointer arithmetic to access contents
  const float z_translation =
      translation_ptr[2]; // NOLINT: Allow pointer arithmetic to access contents

  // translate all point clouds in a batch by the same amount
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      points.index({i, j, POINT_CLOUD_X_IDX}) += x_translation;
      points.index({i, j, POINT_CLOUD_Y_IDX}) += y_translation;
      points.index({i, j, POINT_CLOUD_Z_IDX}) += z_translation;
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

/**
 * Only scale the dimensions of the bounding box (L, H, W) by a constant factor.
 * The labels have the shape (N, M, K), where N is the batch size, M, is the
 * number of labels and K are the features.
 *
 * @param labels are the labels with their bounding boxes.
 * @param factor is the constant factor to scale the box dimensions by.
 */
inline void scale_box_dimensions(at::Tensor labels, float factor) {
  dimensions dims = {labels.size(0), labels.size(1), labels.size(2)};

  // scale all the boxes by the same amount
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      labels.index({i, j, LABEL_W_IDX}) *= factor;
      labels.index({i, j, LABEL_H_IDX}) *= factor;
      labels.index({i, j, LABEL_L_IDX}) *= factor;
    }
  }
}

void translate_random(at::Tensor points, at::Tensor labels, float sigma) {

  std::normal_distribution<float> dist(sigma, 0);

  auto x_translation = std::get<VALUE>(draw_values<float>(dist));
  auto y_translation = std::get<VALUE>(draw_values<float>(dist));
  auto z_translation = std::get<VALUE>(draw_values<float>(dist));

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

void scale_local(at::Tensor point_cloud, at::Tensor labels, float sigma,
                 float max_scale) {

  auto scale_factor =
      get_truncated_normal_value(1, sigma, (1 / max_scale), max_scale);

  dimensions label_dims = {labels.size(0), labels.size(1), labels.size(2)};
  dimensions point_dims = {point_cloud.size(0), point_cloud.size(1),
                           point_cloud.size(2)};

  auto point_indeces =
      torch::zeros({label_dims.num_items, point_dims.num_items}, torch::kI32);

  for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {

    points_in_boxes_cpu(
        labels[i].contiguous(),
        point_cloud[i]
            .index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)})
            .contiguous(),
        point_indeces);

    assert(point_indeces.size(0) == label_dims.num_items);

    for (int j = 0; j < label_dims.num_items; j++) {
      auto box = labels[i][j];
      auto points = point_indeces[j];

      if (!at::any(points).item<bool>()) {
        continue;
      }

      for (int k = 0; k < points.size(0); k++) {
        if (points[k].item<bool>()) {
          point_cloud.index(
              {i, k, torch::indexing::Slice(torch::indexing::None, 3)}) *=
              scale_factor;
        }
      }
    }

    point_indeces.zero_();
  }
  scale_box_dimensions(labels, scale_factor);
}

void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob) {

  auto rng = get_rng();
  std::uniform_int_distribution<std::size_t> distrib(0, HUNDRED_PERCENT - 1);
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

void random_noise(at::Tensor &points, float sigma,
                  const distribution_ranges<float> &ranges, noise_type type) {

  dimensions dims = {points.size(0), points.size(1), points.size(2)};

  auto rng = get_rng();
  std::normal_distribution<float> normal(0.0, sigma);
  std::uniform_real_distribution<float> x_distrib(ranges.x_range.min,
                                                  ranges.x_range.max);
  std::uniform_real_distribution<float> y_distrib(ranges.y_range.min,
                                                  ranges.y_range.max);
  std::uniform_real_distribution<float> z_distrib(ranges.z_range.min,
                                                  ranges.z_range.max);

  const auto num_points = static_cast<std::size_t>(std::abs(normal(rng)));

  // iterate over batches
  for (tensor_size_t batch_num = 0; batch_num < dims.batch_size; batch_num++) {

    const auto x =
        std::get<VECTOR>(draw_values<float>(x_distrib, num_points, true));
    const auto y =
        std::get<VECTOR>(draw_values<float>(y_distrib, num_points, true));
    const auto z =
        std::get<VECTOR>(draw_values<float>(z_distrib, num_points, true));
    const auto i = [type, num_points, min = ranges.uniform_range.min,
                    max = ranges.uniform_range.max]() -> std::vector<float> {
      switch (type) {
      case UNIFORM: {
        std::uniform_real_distribution<float> ud(min, max);
        auto noise_intensity =
            std::get<VECTOR>(draw_values<float>(ud, num_points, true));
        return noise_intensity;
      }
      case SALT_PEPPER: {
        const auto salt_len = num_points / 2;
        const std::vector<float> salt(salt_len, 0);
        const std::vector<float> pepper(num_points - salt_len, MAX_INTENSITY);

        std::vector<float> noise_intensity;
        noise_intensity.reserve(num_points);
        noise_intensity.insert(noise_intensity.begin(), salt.begin(),
                               salt.end());
        noise_intensity.insert(noise_intensity.end(), pepper.begin(),
                               pepper.end());
        return noise_intensity;
      }
      case MIN: {
        std::vector<float> noise_intensity;
        noise_intensity.reserve(num_points);
        std::fill(noise_intensity.begin(), noise_intensity.end(), 0);
        return noise_intensity;
      }
      case MAX: {
        std::vector<float> noise_intensity;
        noise_intensity.reserve(num_points);
        std::fill(noise_intensity.begin(), noise_intensity.end(),
                  MAX_INTENSITY);
        return noise_intensity;
      }

      default:
        // NOTE(tom): This should be unreachable
        assert(false);
      }
    }();

    auto noise_tensor = torch::empty(
        {static_cast<tensor_size_t>(num_points), dims.num_features},
        torch::kF32);

    // NOTE(tom): maybe this can be done more efficiently using masks or by
    // having x, y, z and noise_intensity as tensors from the beginning, but I'd
    // need benchmarks to figure that out

    // 'stack' x, y, z and noise (same as np.stack((x, y, z, noise_intensity),
    // axis=-1))
    for (std::size_t j = 0; j < num_points; j++) {

      noise_tensor[static_cast<tensor_size_t>(j)][POINT_CLOUD_X_IDX] = x[j];
      noise_tensor[static_cast<tensor_size_t>(j)][POINT_CLOUD_Y_IDX] = y[j];
      noise_tensor[static_cast<tensor_size_t>(j)][POINT_CLOUD_Z_IDX] = z[j];
      noise_tensor[static_cast<tensor_size_t>(j)][POINT_CLOUD_I_IDX] = i[j];
    }

    // concatenate points
    points = torch::cat({points, noise_tensor.unsqueeze(0)}, 1);
  }
}

/**
 * Applies a rotation matrix/vector to a batch of points.
 *
 * Expected shape is (b, n, f), where `b` is the batchsize, `n` is the number of
 * items/points and `f` is the number of features.
 *
 * @param points   is the point cloud that is to be rotated.
 * @param rotation is rotation matrix that is used to apply the rotation.
 */
inline void rotate(at::Tensor points, at::Tensor rotation) {

  auto points_vec =
      points.index({torch::indexing::Slice(), torch::indexing::Slice(),
                    torch::indexing::Slice(torch::indexing::None, 3)});

  points.index_put_({torch::indexing::Slice(), torch::indexing::Slice(),
                     torch::indexing::Slice(torch::indexing::None, 3)},
                    torch::matmul(points_vec, rotation));
}

void rotate_deg(at::Tensor points, float angle) {

  auto angle_rad = to_rad(angle);
  auto rotation = rotate_yaw(angle_rad);
  rotate(points, rotation);
}

void rotate_rad(at::Tensor points, float angle) {

  auto rotation = rotate_yaw(angle);
  rotate(points, rotation);
}

void rotate_random(at::Tensor points, at::Tensor labels, float sigma) {

  dimensions point_dims = {points.size(0), points.size(1), points.size(2)};
  auto rot_angle = get_truncated_normal_value(0, sigma, -PI_DEG, PI_DEG);
  auto angle_rad = to_rad(rot_angle);

  auto rotation = rotate_yaw(angle_rad);

  for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < point_dims.num_items; j++) {

      auto points_vec = points.index(
          {i, j, torch::indexing::Slice(torch::indexing::None, 3)});

      points.index_put_(
          {i, j, torch::indexing::Slice(torch::indexing::None, 3)},
          torch::matmul(points_vec, rotation));
    }
  }

  dimensions label_dims = {labels.size(0), labels.size(1), labels.size(2)};
  for (tensor_size_t i = 0; i < label_dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < label_dims.num_items; j++) {
      auto label_vec = labels.index(
          {i, j, torch::indexing::Slice(torch::indexing::None, 3)});

      labels.index_put_(
          {i, j, torch::indexing::Slice(torch::indexing::None, 3)},
          torch::matmul(label_vec, rotation));

      labels[i][j][LABEL_ANGLE_IDX] =
          (labels[i][j][LABEL_ANGLE_IDX] + angle_rad) % (TWO_M_PI);
    }
  }

  // NOTE(tom): coop boxes not implemented
}

[[nodiscard]] torch::Tensor thin_out(at::Tensor points, float sigma) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};

  const auto percent = get_truncated_normal_value(0, sigma, 0, 1);

  const auto num_values = static_cast<tensor_size_t>(
      std::ceil(static_cast<float>(dims.num_items) * (1 - percent)));

  auto new_tensor =
      torch::empty({dims.batch_size, num_values, dims.num_features});

  for (tensor_size_t i = 0; i < dims.batch_size; i++) {

    auto indices = draw_unique_uniform_values<tensor_size_t>(
        static_cast<std::size_t>(dims.num_items),
        static_cast<std::size_t>(num_values));

    for (tensor_size_t j = 0; j < num_values; j++) {
      new_tensor[i][j] = points[i][indices[static_cast<std::size_t>(j)]];
    }
  }

  return new_tensor;
}

[[nodiscard]] std::pair<torch::List<torch::Tensor>, torch::List<torch::Tensor>>
delete_labels_by_min_points(at::Tensor points, at::Tensor labels,
                            at::Tensor names, const tensor_size_t min_points) {

  torch::List<torch::Tensor> batch_labels;
  torch::List<torch::Tensor> batch_names;

  tensor_size_t batch_size = labels.size(0);

  batch_labels.reserve(static_cast<std::size_t>(batch_size));
  batch_names.reserve(static_cast<std::size_t>(batch_size));

  for (tensor_size_t i = 0; i < batch_size; i++) {

    auto [filtered_labels, filtered_names] = _delete_labels_by_min_points(
        points[i], labels[i], names[i], min_points);

    batch_labels.emplace_back(filtered_labels);
    batch_names.emplace_back(filtered_names);
  }

  return {batch_labels, batch_names};
}

/**
 * Applies the `noise_vector` to the `point_vector`.
 *
 * @param point_vector A 3 dimensional point.
 * @param noise_vector A 3 dimensional vector.
 */
constexpr inline void _random_point_noise(f3 &point_vector,
                                          const f3 &noise_vector) noexcept {

  point_vector[0] += noise_vector[0];
  point_vector[1] += noise_vector[1];
  point_vector[2] += noise_vector[2];
}

/**
 * Applies the `noise_value` to every dimension of `point_vector`.
 *
 * @param point_vector A 3 dimensional point.
 * @param noise_vector A 3 dimensional vector.
 */
constexpr inline void _random_point_noise(f3 &point_vector,
                                          const float noise_value) noexcept {
  point_vector[0] += noise_value;
  point_vector[1] += noise_value;
  point_vector[2] += noise_value;
}

void random_point_noise(torch::Tensor points, float sigma) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};

  std::normal_distribution<float> dist(0, sigma);

  // TODO(tom): perf measure this
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      const auto v = points[i][j].data_ptr<float>();
      const auto values = std::get<VECTOR>(draw_values<float>(dist, 3));
      auto point_vector =
          f3{v[0], v[1],
             v[2]}; // NOLINT: Allow pointer arithmetic to access contents
      const auto noise_vector = f3{values[0], values[1], values[2]};

      _random_point_noise(point_vector, noise_vector);
    }
  }
}

void transform_along_ray(torch::Tensor points, float sigma) {
  dimensions dims = {points.size(0), points.size(1), points.size(2)};

  std::normal_distribution<float> dist(0, sigma);

  // TODO(tom): perf measure this
  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {
      const auto v = points[i][j].data_ptr<float>();
      const auto noise = std::get<VALUE>(draw_values<float>(dist));
      auto point_vector =
          f3{v[0], v[1],
             v[2]}; // NOLINT: Allow pointer arithmetic to access contents

      _random_point_noise(point_vector, noise);
    }
  }
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "../include/bindings.hpp"
#else
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

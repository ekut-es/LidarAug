#include "../include/transformations.hpp"
#include "../include/label.hpp"
#include "../include/name.hpp"
#include "../include/stats.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <cmath>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

using Slice = torch::indexing::Slice;
using namespace torch_utils;

void translate(at::Tensor points, const at::Tensor &translation) {
  points.index({Slice(), Slice(), Slice(0, 3)}) += translation;
}

void scale_points(at::Tensor points, const float factor) {
  points.index({Slice(), Slice(), Slice(0, 3)}) *= factor;
}

void scale_labels(at::Tensor labels, const float factor) {
  labels.index({Slice(), Slice(), Slice(0, 6)}) *= factor;
}

/**
 * Only scale the dimensions of the bounding box (L, H, W) by a constant factor.
 * The labels have the shape (N, M, K), where N is the batch size, M, is the
 * number of labels and K are the features.
 *
 * @param labels are the labels with their bounding boxes.
 * @param factor is the constant factor to scale the box dimensions by.
 */
inline void scale_box_dimensions(at::Tensor labels, const float factor) {
  labels.index({Slice(), Slice(), Slice(3, 6)}) *= factor;
}

void translate_random(at::Tensor points, at::Tensor labels, const float sigma) {

  std::normal_distribution<float> dist(sigma, 0);

  const auto x_translation = std::get<VALUE>(draw_values<float>(dist));
  const auto y_translation = std::get<VALUE>(draw_values<float>(dist));
  const auto z_translation = std::get<VALUE>(draw_values<float>(dist));

  const auto translation =
      at::tensor({x_translation, y_translation, z_translation});

  translate(points, translation);
  translate(labels, translation);

  // NOTE(tom): coop boxes not implemented
}

void scale_random(at::Tensor points, at::Tensor labels, const float sigma,
                  const float max_scale) {

  const auto scale_factor =
      get_truncated_normal_value(1, sigma, (1 / max_scale), max_scale);

  scale_points(points, scale_factor);
  scale_labels(labels, scale_factor);

  // NOTE(tom): coop boxes not implemented
}

void scale_local(at::Tensor point_cloud, at::Tensor labels, const float sigma,
                 const float max_scale) {

  const auto scale_factor =
      get_truncated_normal_value(1, sigma, (1 / max_scale), max_scale);

  const dimensions label_dims = {labels.size(0), labels.size(1),
                                 labels.size(2)};
  const dimensions point_dims = {point_cloud.size(0), point_cloud.size(1),
                                 point_cloud.size(2)};

  const auto point_indeces =
      torch::zeros({label_dims.num_items, point_dims.num_items}, torch::kI32);

  for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {

    points_in_boxes_cpu(
        labels[i].contiguous(),
        point_cloud[i].index({Slice(), Slice(0, 3)}).contiguous(),
        point_indeces);

    assert(point_indeces.size(0) == label_dims.num_items);

    for (int j = 0; j < label_dims.num_items; j++) {
      auto points = point_indeces[j];

      if (!at::any(points).item<bool>()) {
        continue;
      }

      for (int k = 0; k < points.size(0); k++) {
        if (points[k].item<bool>()) {
          point_cloud.index({i, k, Slice(torch::indexing::None, 3)}) *=
              scale_factor;
        }
      }
    }

    std::ignore = point_indeces.zero_();
  }
  scale_box_dimensions(labels, scale_factor);
}

void flip_random(at::Tensor points, at::Tensor labels, const std::size_t prob) {

  auto rng = get_rng();
  std::uniform_int_distribution<std::size_t> distrib(0, HUNDRED_PERCENT - 1);

  if (const auto rand = distrib(rng); prob > rand) {
    const dimensions point_dims = {points.size(0), points.size(1),
                                   points.size(2)};

    for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {
      for (tensor_size_t j = 0; j < point_dims.num_items; j++) {
        points.index({i, j, POINT_CLOUD_Y_IDX}) *= -1;
      }
    }

    const dimensions label_dims = {labels.size(0), labels.size(1),
                                   labels.size(2)};
    for (tensor_size_t i = 0; i < label_dims.batch_size; i++) {
      for (tensor_size_t j = 0; j < label_dims.num_items; j++) {
        labels.index({i, j, LABEL_Y_IDX}) *= -1;
        labels.index({i, j, LABEL_ANGLE_IDX}) =
            (labels.index({i, j, LABEL_ANGLE_IDX}) + math_utils::PI_RAD) %
            math_utils::TWO_PI_RAD;
      }
    }
  }
}

[[nodiscard]] torch::Tensor
random_noise(const at::Tensor &points, const float sigma,
             const cpp_utils::distribution_ranges<float> &ranges,
             noise_type type, point_cloud_data::intensity_range max_intensity) {

  const dimensions dims = {points.size(0), points.size(1), points.size(2)};

  auto rng = get_rng();
  std::normal_distribution<float> normal(0.0, sigma);
  std::uniform_real_distribution<float> x_distrib(ranges.x_range.min,
                                                  ranges.x_range.max);
  std::uniform_real_distribution<float> y_distrib(ranges.y_range.min,
                                                  ranges.y_range.max);
  std::uniform_real_distribution<float> z_distrib(ranges.z_range.min,
                                                  ranges.z_range.max);

  const auto num_points = static_cast<std::size_t>(std::abs(normal(rng)));

  auto point_cloud = torch::empty(
      {dims.batch_size, dims.num_items + static_cast<tensor_size_t>(num_points),
       dims.num_features});

  // iterate over batches
  for (tensor_size_t batch_num = 0; batch_num < dims.batch_size; batch_num++) {

    const auto x =
        std::get<VECTOR>(draw_values<float>(x_distrib, num_points, true));
    const auto y =
        std::get<VECTOR>(draw_values<float>(y_distrib, num_points, true));
    const auto z =
        std::get<VECTOR>(draw_values<float>(z_distrib, num_points, true));
    const auto i = [type, num_points, min = ranges.uniform_range.min,
                    max = ranges.uniform_range.max,
                    max_intensity]() -> std::vector<float> {
      switch (type) {
      case noise_type::UNIFORM: {
        std::uniform_real_distribution<float> ud(min, max);
        auto noise_intensity =
            std::get<VECTOR>(draw_values<float>(ud, num_points, true));
        return noise_intensity;
      }
      case noise_type::SALT_PEPPER: {
        const auto salt_len = num_points / 2;
        const std::vector<float> salt(salt_len, 0);
        const std::vector<float> pepper(num_points - salt_len,
                                        static_cast<float>(max_intensity));

        std::vector<float> noise_intensity;
        noise_intensity.reserve(num_points);
        noise_intensity.insert(noise_intensity.begin(), salt.begin(),
                               salt.end());
        noise_intensity.insert(noise_intensity.end(), pepper.begin(),
                               pepper.end());
        return noise_intensity;
      }
      case noise_type::MIN: {
        std::vector<float> noise_intensity;
        noise_intensity.reserve(num_points);
        std::fill(noise_intensity.begin(), noise_intensity.end(), 0);
        return noise_intensity;
      }
      case noise_type::MAX: {
        std::vector<float> noise_intensity;
        noise_intensity.reserve(num_points);
        std::fill(noise_intensity.begin(), noise_intensity.end(),
                  static_cast<float>(max_intensity));
        return noise_intensity;
      }

      default:
        // NOTE(tom): This should be unreachable
        assert(false);
      }
    }();

    auto noise_tensor = torch::empty(
        {static_cast<tensor_size_t>(num_points), dims.num_features}, F32);

    // NOTE(tom): maybe this can be done more efficiently using masks or by
    // having x, y, z and noise_intensity as tensors from the beginning, but I'd
    // need benchmarks to figure that out

    // 'stack' x, y, z and noise (same as np.stack((x, y, z, noise_intensity),
    // axis=-1))
    for (std::size_t j = 0; j < num_points; j++) {

      noise_tensor.index_put_(
          {static_cast<tensor_size_t>(j), POINT_CLOUD_X_IDX}, x[j]);
      noise_tensor.index_put_(
          {static_cast<tensor_size_t>(j), POINT_CLOUD_Y_IDX}, y[j]);
      noise_tensor.index_put_(
          {static_cast<tensor_size_t>(j), POINT_CLOUD_Z_IDX}, z[j]);
      noise_tensor.index_put_(
          {static_cast<tensor_size_t>(j), POINT_CLOUD_I_IDX}, i[j]);
    }

    // concatenate points
    point_cloud = torch::cat({points, noise_tensor.unsqueeze(0)}, 1);
  }

  return point_cloud;
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
inline void rotate(at::Tensor points, const at::Tensor &rotation) {

  const auto points_vec =
      points.index({Slice(), Slice(), Slice(torch::indexing::None, 3)});

  points.index_put_({Slice(), Slice(), Slice(torch::indexing::None, 3)},
                    torch::matmul(points_vec, rotation));
}

void rotate_deg(at::Tensor points, const float angle) {

  const auto angle_rad = math_utils::to_rad(angle);
  const auto rotation = math_utils::rotate_yaw(angle_rad);
  rotate(points, rotation);
}

void rotate_rad(at::Tensor points, const float angle) {

  const auto rotation = math_utils::rotate_yaw(angle);
  rotate(points, rotation);
}

void rotate_random(at::Tensor points, at::Tensor labels, const float sigma) {

  const dimensions point_dims = {points.size(0), points.size(1),
                                 points.size(2)};
  const auto rot_angle = get_truncated_normal_value(
      0, sigma, -math_utils::PI_DEG, math_utils::PI_DEG);
  const auto angle_rad = math_utils::to_rad(rot_angle);

  const auto rotation = math_utils::rotate_yaw(angle_rad);

  for (tensor_size_t i = 0; i < point_dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < point_dims.num_items; j++) {

      auto points_vec = points.index({i, j, Slice(torch::indexing::None, 3)});

      points.index_put_({i, j, Slice(torch::indexing::None, 3)},
                        torch::matmul(points_vec, rotation));
    }
  }

  const dimensions label_dims = {labels.size(0), labels.size(1),
                                 labels.size(2)};
  for (tensor_size_t i = 0; i < label_dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < label_dims.num_items; j++) {
      auto label_vec = labels.index({i, j, Slice(torch::indexing::None, 3)});

      labels.index_put_({i, j, Slice(torch::indexing::None, 3)},
                        torch::matmul(label_vec, rotation));

      labels[i][j][LABEL_ANGLE_IDX] =
          (labels[i][j][LABEL_ANGLE_IDX] + angle_rad) %
          (math_utils::TWO_PI_RAD);
    }
  }

  // NOTE(tom): coop boxes not implemented
}

[[nodiscard]] torch::Tensor thin_out(const at::Tensor &points,
                                     const float sigma) {
  const dimensions dims = {points.size(0), points.size(1), points.size(2)};

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
      new_tensor.index_put_({i, j},
                            points[i][indices[static_cast<std::size_t>(j)]]);
    }
  }

  return new_tensor;
}

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
delete_labels_by_min_points(const at::Tensor &points, const at::Tensor &labels,
                            const at::Tensor &names,
                            const tensor_size_t min_points) {
  const tensor_size_t batch_size = labels.size(0);

  std::vector<torch::Tensor> labels_list;
  std::vector<torch::Tensor> names_list;

  labels_list.reserve(static_cast<std::size_t>(batch_size));
  names_list.reserve(static_cast<std::size_t>(batch_size));

  for (tensor_size_t i = 0; i < batch_size; i++) {

    auto [filtered_labels, filtered_names] = delete_labels_by_min_points_(
        points[i], labels[i], names[i], min_points, i);

    labels_list.emplace_back(filtered_labels);
    names_list.emplace_back(filtered_names);
  }

  auto batch_labels =
      torch::stack(labels_list).reshape({-1, LABEL_NUM_FEATURES + 1});
  auto batch_names =
      torch::stack(names_list).reshape({-1, NAME_NUM_FEATURES + 1});

  return std::make_pair(batch_labels, batch_names);
}

void random_point_noise(torch::Tensor points, const float sigma) {
  const dimensions dims = {points.size(0), points.size(1), points.size(2)};

  const auto noise =
      torch::normal(0, sigma, {dims.batch_size, dims.num_items, 3});

  points.index({torch::indexing::Slice(), torch::indexing::Slice(),
                torch::indexing::Slice(0, 3)}) += noise;
}

void transform_along_ray(torch::Tensor points, const float sigma) {
  const dimensions dims = {points.size(0), points.size(1), points.size(2)};

  const auto noise =
      torch::normal(0, sigma, {dims.batch_size, dims.num_items, 1})
          .repeat({1, 1, 3});

  points.index({torch::indexing::Slice(), torch::indexing::Slice(),
                torch::indexing::Slice(0, 3)}) += noise;
}

void intensity_noise(torch::Tensor points, const float sigma,
                     const point_cloud_data::intensity_range max_intensity) {
  const dimensions dims = {points.size(0), points.size(1), points.size(2)};

  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {

      const float max_shift = static_cast<float>(max_intensity) -
                              points[i][j][POINT_CLOUD_I_IDX].item<float>();

      const float intensity_shift =
          get_truncated_normal_value(0, sigma, 0, max_shift);

      points[i][j][POINT_CLOUD_I_IDX] += intensity_shift;
    }
  }
}

void intensity_shift(torch::Tensor points, const float sigma,
                     const point_cloud_data::intensity_range max_intensity) {
  const float intensity_shift = get_truncated_normal_value(
      0, sigma, 0, static_cast<float>(max_intensity));

  const dimensions dims = {points.size(0), points.size(1), points.size(2)};

  for (tensor_size_t i = 0; i < dims.batch_size; i++) {
    for (tensor_size_t j = 0; j < dims.num_items; j++) {

      const auto current_intensity =
          points[i][j][POINT_CLOUD_I_IDX].item<float>();
      const float new_intensity = std::min(current_intensity + intensity_shift,
                                           static_cast<float>(max_intensity));
      points.index_put_({i, j, POINT_CLOUD_I_IDX}, new_intensity);
    }
  }
}

[[nodiscard]] torch::Tensor
local_to_world_transform(const torch::Tensor &lidar_pose) {

  auto transformation = torch::eye(4);

  // translations
  transformation.index_put_({0, 3}, lidar_pose[0]);
  transformation.index_put_({1, 3}, lidar_pose[1]);
  transformation.index_put_({2, 3}, lidar_pose[2]);

  // rotations
  const auto cos_roll = lidar_pose[3].deg2rad().cos();
  const auto sin_roll = lidar_pose[3].deg2rad().sin();
  const auto cos_yaw = lidar_pose[4].deg2rad().cos();
  const auto sin_yaw = lidar_pose[4].deg2rad().sin();
  const auto cos_pitch = lidar_pose[5].deg2rad().cos();
  const auto sin_pitch = lidar_pose[5].deg2rad().sin();

  transformation.index_put_({2, 0}, sin_pitch);

  transformation.index_put_({0, 0}, cos_pitch * cos_yaw);
  transformation.index_put_({1, 0}, sin_yaw * cos_pitch);
  transformation.index_put_({2, 1}, -cos_pitch * sin_roll);
  transformation.index_put_({2, 2}, cos_pitch * cos_roll);

  transformation.index_put_({0, 1}, cos_yaw * sin_pitch * sin_roll -
                                        sin_yaw * cos_roll);
  transformation.index_put_({0, 2}, -cos_yaw * sin_pitch * cos_roll -
                                        sin_yaw * sin_roll);
  transformation.index_put_({1, 1}, sin_yaw * sin_pitch * sin_roll +
                                        cos_yaw * cos_roll);
  transformation.index_put_({1, 2}, -sin_yaw * sin_pitch * cos_roll +
                                        cos_yaw * sin_roll);

  return transformation;
}

[[nodiscard]] torch::Tensor
local_to_local_transform(const torch::Tensor &from_pose,
                         const torch::Tensor &to_pose) {

  const auto local_to_world = local_to_world_transform(from_pose);
  const auto world_to_local =
      torch::linalg_inv(local_to_world_transform(to_pose));

  return world_to_local.mm(local_to_world);
}

void apply_transformation(torch::Tensor points,
                          const torch::Tensor &transformation_matrix) {

  // Extract x, y, z coordinates
  const auto coords = points.index({Slice(), Slice(), Slice(0, 3)});

  // apply transformation
  const auto transformed_points =
      coords.matmul(transformation_matrix.permute({1, 0}));

  // Update x, y, z in place
  points.index_put_({Slice(), Slice(), Slice(0, 3)}, transformed_points);
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "../include/transformation_bindings.hpp"
#else
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

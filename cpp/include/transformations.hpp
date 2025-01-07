
#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include "../include/point_cloud.hpp"
#include "../include/roiware_utils.h"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include <torch/serialize/tensor.h>

enum struct noise_type : std::uint8_t { UNIFORM, SALT_PEPPER, MIN, MAX };

void translate(at::Tensor points, const at::Tensor &translation);
void scale_points(at::Tensor points, float factor);
void scale_labels(at::Tensor labels, float factor);

void translate_random(at::Tensor points, at::Tensor labels, float sigma);

/**
 * Scales the points and labels by a random factor.
 * This factor is drawn from a truncated normal distribution.
 * The truncated normal distribution has a mean of 1. The standard deviation, as
 * well as upper and lower limits are determined by the function parameters.
 *
 * @param points    is the point cloud that contains the points that will be
 *                  scaled.
 * @param labels    are the labels belonging to the aforementioned point cloud.
 * @param sigma     is the the standard deviation of the truncated normal
 *                  distribution.
 * @param max_scale is the upper limit of the truncated normal distribution. The
 *                  lower limit is the inverse.
 */
void scale_random(at::Tensor points, at::Tensor labels, float sigma,
                  float max_scale);
/**
 * Scales the points that are part of a box and the corresponding labels by a
 * random factor.
 *
 * This factor is drawn from a truncated normal distribution.
 * The truncated normal distribution has a mean of 1. The standard deviation, as
 * well as upper and lower limits are determined by the function parameters.
 *
 * @param point_cloud is the point cloud that contains the points that will be
 *                    scaled.
 * @param labels      are the labels belonging to the aforementioned point
 *                    cloud.
 * @param sigma       is the the standard deviation of the truncated normal
 *                    distribution.
 * @param max_scale   is the upper limit of the truncated normal distribution.
 *                    The lower limit is the inverse.
 */
void scale_local(at::Tensor point_cloud, at::Tensor labels, float sigma,
                 float max_scale);
void flip_random(at::Tensor points, at::Tensor labels, std::size_t prob);

/**
 * Rotates a batch of points anlong the 'z' axis (yaw).
 *
 * @param points is the point cloud that the rotation is applied to.
 * @param angle  is the angle (in degrees) by which the points are to be
 * rotated.
 */
void rotate_deg(at::Tensor points, float angle);

/**
 * Rotates a batch of points anlong the 'z' axis (yaw).
 *
 * @param points is the point cloud that the rotation is applied to.
 * @param angle  is the angle (in radians) by which the points are to be
 *               rotated.
 */
void rotate_rad(at::Tensor points, float angle);

/**
 * Introduces random noise to a point cloud.
 *
 * @param points          is a (n, 4) tensor representing the point cloud.
 * @param sigma           is the standard deviation of the normal distribution
 *                        that is used to draw the number of points to be added.
 * @param ranges          are the boundaries in (min and max (x, y, z) values)
 *                        in which the new points can be created.
 * @param type            is the type of noise that is to be introduced.
 * @param max_intensity   is the maximum intensity value in the dataset.
 *
 * @returns a new tensor with the new points added.
 */
[[nodiscard]] torch::Tensor
random_noise(const at::Tensor &points, float sigma,
             const cpp_utils::distribution_ranges<float> &ranges,
             noise_type type, point_cloud_data::intensity_range max_intensity);

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
[[nodiscard]] torch::Tensor thin_out(const at::Tensor &points, float sigma);

void rotate_random(at::Tensor points, at::Tensor labels, float sigma);

/**
 * Checks the amount of points for each bounding box.
 * If the number of points is smaller than a given threshold, the box is removed
 * along with its label.
 *
 * @param points     is the point_cloud.
 * @param labels     are the bounding boxes of objects.
 * @param names      are the names/labels of these boxes.
 * @param min_points is the point threshold.
 *
 * @returns a `std::pair` of `torch::Tensor` containing the new labels and new
 * names (in that order) as a flat tensor with the batch index they were in as
 * an additional feature.
 */
[[nodiscard]] std::pair<torch::Tensor, torch::Tensor>
delete_labels_by_min_points(const at::Tensor &points, const at::Tensor &labels,
                            const at::Tensor &names, tensor_size_t min_points);

/**
 * Checks the amount of points for each bounding box.
 * If the number of points is smaller than a given threshold, the box is removed
 * along with its label.
 * This function function expectes all tensors in the shape of (n, m), where m
 * is the number of features and n is the number of elements.
 * It does not handle batches, but adds the batch index to the point.
 *
 * @param points     is the point_cloud.
 * @param labels     are the bounding boxes of objects.
 * @param names      are the names/labels of these boxes.
 * @param min_points is the point threshold.
 * @param batch_idx  is the index of the batch being currently processed.
 *
 * @returns a `std::pair` of `torch::Tensor` containing the new labels and their
 *          names (in that order), as well as the batch index.
 */
[[nodiscard]] inline std::pair<torch::Tensor, torch::Tensor>
delete_labels_by_min_points_(const at::Tensor &points, const at::Tensor &labels,
                             const at::Tensor &names,
                             const tensor_size_t min_points,
                             const tensor_size_t batch_idx) {

  const tensor_size_t num_labels = labels.size(0);
  const tensor_size_t num_points = points.size(0);

  const tensor_size_t label_features = labels.size(1);
  const tensor_size_t name_features = names.size(1);

  const auto point_indices =
      torch::zeros({num_labels, num_points}, torch::kI32);
  points_in_boxes_cpu(
      labels.contiguous(),
      points.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)})
          .contiguous(),
      point_indices);

  assert(point_indices.size(0) == num_labels);

  const auto not_deleted_indices = point_indices.sum(1).ge(min_points);

  const auto not_deleted_labels =
      labels.index({not_deleted_indices.nonzero().squeeze()})
          .view({-1, label_features});
  const auto not_deleted_names =
      names.index({not_deleted_indices.nonzero().squeeze()})
          .view({-1, name_features});

  // create tensors for the index
  const auto idx_labels =
      torch::empty({not_deleted_labels.size(0)}, labels.options().dtype());
  const auto idx_names =
      torch::empty({not_deleted_names.size(0)}, names.options().dtype());

  std::ignore = idx_labels.fill_(batch_idx);
  std::ignore = idx_names.fill_(batch_idx);

  // merge filtered tensors with index
  auto merged_labels =
      torch::cat({not_deleted_labels, idx_labels.reshape({1, -1})}, 1);
  auto merged_names =
      torch::cat({not_deleted_names, idx_names.reshape({1, -1})}, 1);

  return {merged_labels, merged_names};
}

void random_point_noise(torch::Tensor points, float sigma);

void transform_along_ray(torch::Tensor points, float sigma);

void intensity_noise(torch::Tensor points, float sigma,
                     point_cloud_data::intensity_range max_intensity);

void intensity_shift(torch::Tensor points, float sigma,
                     point_cloud_data::intensity_range max_intensity);

/**
 * Creates a transformation matrix from the local system into the global
 * coordinate frame.
 *
 * @param lidar_pose is the local coordinate frame (x, y, z, roll, yaw, pitch).
 * @returns the homogeneous transformation matrix into the global coordinate
 *           frame.
 */
[[nodiscard]] torch::Tensor
local_to_world_transform(const torch::Tensor &lidar_pose);

/**
 * Creates a transformation matrix from the local system into a 'target'
 * coordinate frame.
 *
 * @param from_pose is the local coordinate frame (x, y, z, roll, yaw, pitch).
 * @param to_pose   is the target coordinate frame (x, y, z, roll, yaw, pitch).
 * @returns  the homogeneous transformation matrix into the target coordinate
 *           frame.
 */
[[nodiscard]] torch::Tensor
local_to_local_transform(const torch::Tensor &from_pose,
                         const torch::Tensor &to_pose);

/**
 * Applies a transformation matrix to an entire point cloud with the shape (B,
 * N, F), where B is the number of batches and N is the number of points.
 *
 * @param points                is the point cloud that the transformation
 *                              matrix is applied to.
 * @param transformation_matrix is the transformation matrix.
 */
void apply_transformation(torch::Tensor points,
                          const torch::Tensor &transformation_matrix);

#endif // !TRANSFORMATIONS_HPP

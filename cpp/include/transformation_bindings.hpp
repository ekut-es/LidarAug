
#include "../include/tensor.hpp"
#include "../include/transformations.hpp"
#include "../include/utils.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

using arg = pybind11::arg;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::enum_<noise_type>(m, "NoiseType",
                              "Indicates how the noise is added:")
      .value("UNIFORM", noise_type::UNIFORM,
             "The noise values are drawn from a uniform distribution.")
      .value("SALT_PEPPER", noise_type::SALT_PEPPER,
             "Half of the added values have the maximum intensity, the other "
             "half the minimum intensity.")
      .value("MIN", noise_type::MIN,
             "The noise values are equal to the minimum intensity.")
      .value("MAX", noise_type::MAX,
             "The noise values are equal to the maximum intensity.");

  // NOTE(tom): Unfortunately it is necessary to export this with defined types,
  //            as PyBind does not appear to support generics/templates.
  pybind11::class_<cpp_utils::range<float>>(m, "DistributionRange")
      .def(pybind11::init<float, float>(), arg("min"), arg("max"));

  pybind11::class_<cpp_utils::distribution_ranges<float>>(m,
                                                          "DistributionRanges")
      .def(pybind11::init<cpp_utils::range<float>, cpp_utils::range<float>,
                          cpp_utils::range<float>, cpp_utils::range<float>>(),
           arg("x_range"), arg("y_range"), arg("z_range"), arg("uniform_range"))
      .def(pybind11::init<cpp_utils::range<float>, cpp_utils::range<float>,
                          cpp_utils::range<float>>(),
           arg("x_range"), arg("y_range"), arg("z_range"));

  m.def("translate", &translate, arg("points"), arg("translation"),
        "Moves batches of points by a specific amount.\n"
        "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
        "the number of batches, `N` is the number of points in each batch and "
        "`F` is the number of features. For this transformation, only the "
        "first 3 features `(x, y, z)` are used.\n"
        "\n"
        ":param points:      is the point cloud with the points are to be "
        "moved.\n"
        ":param translation: is the translation vector that specifies by how "
        "much they points are moved.\n");

  m.def(
      "translate_random", &translate_random, arg("points"), arg("labels"),
      arg("sigma"),
      "Generates a random (3D) translation vector using a normal distribution "
      "and applies it to all the points and labels.\n"
      "The `points` and `labels` are expected to have the shape `(B, N, F)`, "
      "where `B` is the number of batches, `N` is the number of points in each "
      "batch and `F` is the number of features. For this transformation, only "
      "the first 3 features `(x, y, z)` are used.\n"
      "\n"
      ":param points: is the point cloud with the points that are translated.\n"
      ":param labels: are the labels belonging to the aforementioned point "
      "cloud.\n"
      ":param sigma:  is the standard deviation of the normal distribution.\n");

  m.def("scale_points", &scale_points, arg("points"), arg("scaling_factor"),
        "Scales points by a constant factor.\n"
        "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
        "the number of batches, `N` is the number of points in each batch and "
        "`F` is the number of features. For this transformation, only the "
        "first 3 features `(x, y, z)` are used.\n"
        "\n"
        ":param points:         is the point cloud whose points are scaled.\n"
        ":param scaling_factor: is the factor that the (x, y, z) coordinates "
        "are multiplied by.\n");

  m.def("rotate_deg", &rotate_deg, arg("points"), arg("angle"),
        "Rotates a batch of points along the 'z' axis (yaw).\n"
        "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
        "the number of batches, `N` is the number of points in each batch and "
        "`F` is the number of features.\n"
        "\n"
        ":param points: is the point cloud that the rotation is applied to.\n"
        ":param angle:  is the angle (in degrees) by which the points are to "
        "be rotated.\n");

  m.def("rotate_rad", &rotate_rad, arg("points"), arg("angle"),
        "Rotates a batch of points along the 'z' axis (yaw).\n"
        "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
        "the number of batches, `N` is the number of points in each batch and "
        "`F` is the number of features.\n"
        "\n"
        ":param points: is the point cloud that the rotation is applied to.\n"
        ":param angle:  is the angle (in radians) by which the points are to "
        "be rotated.\n");

  m.def(
      "scale_random", &scale_random, arg("points"), arg("labels"), arg("sigma"),
      arg("max_scale"),
      "Scales the points and labels by a random factor.\n"
      "This factor is drawn from a truncated normal distribution.\n"
      "The truncated normal distribution has a mean of 1. The standard "
      "deviation, as well as upper and lower limits are determined by the "
      "function parameters.\n"
      "The `points` and `labels` are expected to have the shape `(B, N, F)`, "
      "where `B` is the number of batches, `N` is the number of points in each "
      "batch and `F` is the number of features. For this transformation, "
      "the first 3 features `(x, y, z)` of the `points` and the first 6 of "
      "`labels` `(x, y, z, w, h, l)` are used.\n"
      "\n"
      ":param points:    is the point cloud that contains the points that "
      "will be scaled.\n"
      ":param labels:    are the labels belonging to the aforementioned "
      "point cloud.\n"
      ":param sigma:     is the standard deviation of the truncated normal "
      "distribution.\n"
      ":param max_scale: is the upper limit of the truncated normal "
      "distribution. The lower limit is the inverse.\n");

  m.def(
      "scale_local", &scale_local, arg("points"), arg("labels"), arg("sigma"),
      arg("max_scale"),
      "Scales the points that are part of a box and the corresponding labels "
      "by a random factor.\n"
      "\n"
      "This factor is drawn from a truncated normal distribution.\n"
      "The truncated normal distribution has a mean of 1. The standard "
      "deviation, as well as upper and lower limits are determined by the "
      "function parameters.\n"
      "The `points` and `labels` are expected to have the shape `(B, N, F)`, "
      "where `B` is the number of batches, `N` is the number of points in each "
      "batch and `F` is the number of features. For this transformation, "
      "the first 3 features `(x, y, z)` of the `points` and the features 4 to "
      "6 of `labels` `(_, _, _, w, h, l)` are used.\n"
      "\n"
      ":param points:    is the point cloud that contains the points that "
      "will be scaled.\n"
      ":param labels:    are the labels belonging to the aforementioned "
      "point cloud.\n"
      ":param sigma:     is the standard deviation of the truncated normal "
      "distribution.\n"
      ":param max_scale: is the upper limit of the truncated normal "
      "distribution. The lower limit is the inverse.\n");

  m.def(
      "flip_random", &flip_random, arg("points"), arg("labels"), arg("prob"),
      "Flips all the points in the point cloud with a probability of `prob` % "
      "in the direction of the y-axis.\n"
      "The `points` and `labels` are expected to have the shape `(B, N, F)`, "
      "where `B` is the number of batches, `N` is the number of points in each "
      "batch and `F` is the number of features. This transformation only "
      "uses the `points` coordinates as well as the coordinates and angle of "
      "the `labels`.\n"
      "\n"
      ":param points:  is the point cloud containing the points that will be "
      "flipped.\n"
      ":param labels:  are the corresponding labels.\n"
      ":param prob:    is the probability with which the points should be "
      "flipped.\n");

  m.def(
      "rotate_random", &rotate_random, arg("points"), arg("labels"),
      arg("sigma"),
      "Rotates points and labels.\n"
      "The number of degrees that they are rotated by is determined by a "
      "randomly generated value from a normal distribution.\n"
      "The `points` and `labels` are expected to have the shape `(B, N, F)`, "
      "where `B` is the number of batches, `N` is the number of points in each "
      "batch and `F` is the number of features. This transformation only "
      "uses the `points` coordinates as well as the coordinates and angle of "
      "the `labels`.\n"
      "\n"
      ":param points: is the point cloud that the rotation is applied to.\n"
      ":param labels: are the labels belonging to the point cloud that the "
      "rotation is applied to.\n"
      ":param sigma:  is the standard deviation of the normal distribution.\n");

  m.def(
      "thin_out", &thin_out, arg("points"), arg("sigma"),
      "Randomly generates a percentage from a norma distribution, which "
      "determines how many items should be 'thinned out'. From that percentage "
      "random indices are uniformly drawn (in a random order, where each index "
      "is unique).\n"
      "\n"
      "Finally, a new tensor is created containing the items present at those\n"
      "indices.\n"
      "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
      "the number of batches, `N` is the number of points in each batch and "
      "`F` is the number of features.\n"
      "\n"
      ":param points: is the point cloud.\n"
      ":param sigma:  is the standard deviation of the distribution that "
      "generates the percentage.\n"
      ":return: a new tensor with the shape `(B, M, F)` containing the new set "
      "of points.\n");

  m.def(
      "random_noise", &random_noise, arg("points"), arg("sigma"), arg("ranges"),
      arg("noise_type"), arg("max_intensity"),
      "Adds random amount of points (drawn using a normal distribution) at "
      "random coordinates (within predetermined ranges) with a random "
      "intensity according to specific noise type.\n"
      "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
      "the number of batches, `N` is the number of points in each batch and "
      "`F` is the number of features, with `F` = 4; `(x, y, z, i)`.\n"
      "\n"
      ":param points:         is the point cloud that the points will be "
      "added to.\n"
      ":param sigma:          is the standard deviation of the normal "
      "distribution that is used to draw the number of points to be added.\n"
      ":param ranges:         are the boundaries in (min and max (x, y, z) "
      "values) in which the new points can be created.\n"
      ":param noise_type:     is one of a number of 'patterns' that can be "
      "used to generate the points.\n"
      ":param max_intensity:  is the maximum intensity value in the "
      "dataset.\n"
      ":return: a new tensor with the shape `(B, M, F)` containing the points "
      "of the input tensor `points` and the newly generated points.\n");

  m.def("delete_labels_by_min_points", &delete_labels_by_min_points,
        arg("points"), arg("labels"), arg("names"), arg("min_points"),
        "Checks the amount of points for each bounding box.\n"
        "If the number of points is smaller than a given threshold, the box is "
        "removed along with its label.\n"
        "\n"
        ":param points:     is the point_cloud.\n"
        ":param labels:     are the bounding boxes of objects.\n"
        ":param names:      are the names/labels of these boxes.\n"
        ":param min_points: is the point threshold.\n"
        ":return: The batch with the new labels and the batch with the new "
        "names.\n",
        pybind11::return_value_policy::reference_internal);

  m.def(
      "random_point_noise", &random_point_noise, arg("points"), arg("sigma"),
      "Moves each point in the point cloud randomly.\n"
      "How much each coordinate is changed is decided by values drawn from a "
      "normal distribution.\n"
      "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
      "the number of batches, `N` is the number of points in each batch and "
      "`F` is the number of features.\n"
      "This transformation only uses the first 3 features `(x, y, z)` of "
      "`points`.\n"
      "\n"
      ":param points: is the point cloud from which each point is moved.\n"
      ":param sigma:  is the standard deviation of the normal distribution.\n");

  m.def(
      "transform_along_ray", &transform_along_ray, arg("points"), arg("sigma"),
      "Moves each point in the point cloud randomly along a ray.\n"
      "How much it is moved is decided by a value drawn from a normal "
      "distribution.\n"
      "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
      "the number of batch, `N` is the number of points in each batch and `F` "
      "is the number of features."
      "This transformation only uses the first 3 features `(x, y, z)` of "
      "`points`.\n"
      "\n"
      ":param points: is the point cloud from which each point is moved.\n"
      ":param sigma:  is the standard deviation of the normal distribution.\n");

  m.def(
      "intensity_noise", &intensity_noise, arg("points"), arg("sigma"),
      arg("max_intensity"),
      "Shifts the intensity value of every point in the point cloud by a "
      "random amount drawn from a normal distribution.\n"
      "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
      "the number of batch, `N` is the number of points in each batch and `F` "
      "is the number of features."
      "This transformation only uses the fourth feature 'intensity' of "
      "`points`.\n"
      "\n"
      ":param points:        is the point cloud with all the points.\n"
      ":param sigma:         is the standard deviation of the normal "
      "distribution.\n"
      ":param max_intensity: is the maximum intensity value (either 1 or "
      "255, depending on the dataset).\n");

  m.def(
      "intensity_shift", &intensity_shift, arg("points"), arg("sigma"),
      arg("max_intensity"),
      "Shifts the intensity value of every point in the point cloud by a "
      "single value drawn from a normal distribution.\n"
      "The `points` are expected to have the shape `(B, N, F)`, where `B` is "
      "the number of batch, `N` is the number of points in each batch and `F` "
      "is the number of features."
      "This transformation only uses the fourth feature 'intensity' of "
      "`points`.\n"
      "\n"
      ":param points:        is the point cloud with all the points.\n"
      ":param sigma:         is the standard deviation of the normal "
      "distribution.\n"
      ":param max_intensity: is the maximum intensity value (either 1 or "
      "255, depending on the dataset).\n");

  m.def("local_to_world_transform", &local_to_world_transform,
        arg("lidar_pose"),
        "Creates a transformation matrix from the local system into the global "
        "coordinate frame.\n"
        "The expected input shape is `(6)` -> `(x, y, z, roll, yaw, pitch)`.\n"
        "\n"
        ":param lidar_pose: is the local coordinate frame (x, y, z, roll, yaw, "
        "pitch).\n"
        ":return: the homogeneous transformation matrix into the global "
        "coordinate frame.\n");

  m.def("local_to_local_transform", &local_to_local_transform, arg("from_pose"),
        arg("to_pose"),
        "Creates a transformation matrix from the local system into a 'target' "
        "coordinate frame.\n"
        "The expected input shape for both tensors is `(6)` -> `(x, y, z, "
        "roll, yaw, pitch)`.\n"
        "\n"
        ":param from_pose: is the local coordinate frame (x, y, z, roll, yaw, "
        "pitch).\n"
        ":param to_pose:   is the target coordinate frame (x, y, z, roll, yaw, "
        "pitch).\n"
        ":return: the homogeneous transformation matrix into the target "
        "coordinate frame.\n");

  m.def(
      "apply_transformation", &apply_transformation, arg("points"),
      arg("transformation_matrix"),
      "Applies a transformation matrix to an entire point cloud with the shape "
      "`(B, N, F)`, where `B` is the number of batches, `N` is the number of "
      "points and `F` has the size 4 and shape `[x, y, z, i]`.\n"
      "\n"
      ":param points:                is the point cloud that the "
      "transformation\n"
      "                              matrix is applied to.\n"
      ":param transformation_matrix: is the transformation matrix.\n");

  m.def("change_sparse_representation", &change_sparse_representation,
        arg("input"), arg("batch_idx"),
        "Changes the representation of a sparse tensor from a flat 2D tensor "
        "`(N, F)`, where `F` is the number of features to a 3D tensor `(B, n, "
        "f)`, where `B` is the number of batches, `n` is the number of tensors "
        "in each batch and `f` is the number of features (equal to F-1).\n"
        "0s are used for padding.\n"
        "\n"
        ":param input:     is the input tensor.\n"
        ":param batch_idx: is the index of the batch index.\n"
        "\n"
        ":return: a new tensor with 0s for padding.\n");
}

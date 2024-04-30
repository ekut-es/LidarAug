#ifndef UTILS_HPP
#define UTILS_HPP

#include "tensor.hpp"
#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/detail/intersection/interface.hpp>
#include <boost/geometry/algorithms/detail/intersects/interface.hpp>
#include <boost/geometry/algorithms/union.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <cmath>
#include <execution>
#include <torch/serialize/tensor.h>
#include <vector>

namespace math_utils {

constexpr float PI_DEG = 180.0;
constexpr float PI_RAD = static_cast<float>(M_PI);
constexpr float TWO_PI_RAD = 2.0f * PI_RAD;

/**
 * Generates a rotation matrix around the 'z' axis (yaw) from the provided
 * angle.
 *
 * @param angle is the angle (in radians)
 *
 * @returns a 3x3 rotation matrix (in form of a torch::Tensor)
 */
[[nodiscard]] inline torch::Tensor rotate_yaw(float angle) {

  float cos_angle = cos(angle);
  float sin_angle = sin(angle);

  auto rotation = torch::tensor({{cos_angle, 0.0f, sin_angle},
                                 {0.0f, 1.0f, 0.0f},
                                 {-sin_angle, 0.0f, cos_angle}});
  return rotation;
}

[[nodiscard]] constexpr inline float to_rad(float angle) noexcept {
  return angle * (PI_RAD / PI_DEG);
}

[[nodiscard]] inline double
compute_condition_number(const torch::Tensor &matrix) {

  // Perform Singular Value Decomposition (SVD)
  const auto svd_result = torch::svd(matrix);
  const auto singular_values = std::get<1>(svd_result);

  // Compute the condition number
  const auto max_singular_value = singular_values.max().item<double>();
  const auto min_singular_value = singular_values.min().item<double>();
  const double condition_number = max_singular_value / min_singular_value;

  return condition_number;
}

} // namespace math_utils

namespace torch_utils {
constexpr auto F32 = torch::kF32;
constexpr auto F64 = torch::kF64;
} // namespace torch_utils

namespace evaluation_utils {

typedef boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian>
    point_t;
typedef boost::geometry::model::polygon<point_t> polygon_t;
typedef boost::geometry::model::multi_polygon<polygon_t> multi_polygon_t;

[[nodiscard]] inline std::vector<polygon_t>
convert_format(const torch::Tensor &boxes) {

  std::vector<polygon_t> ps;

  ps.resize(static_cast<std::size_t>(boxes.size(0)));
  for (tensor_size_t i = 0; i < boxes.size(0); i++) {
    auto box = boxes[i];

    point_t p1{box[0][0].item<float>(), box[0][1].item<float>()};
    point_t p2{box[1][0].item<float>(), box[1][1].item<float>()};
    point_t p3{box[2][0].item<float>(), box[2][1].item<float>()};
    point_t p4{box[3][0].item<float>(), box[3][1].item<float>()};

    polygon_t p{{p1, p2, p3, p4}};

    ps.emplace_back(p);
  }

  return ps;
}

/**
 * Computes intersection over union between `box` and `boxes`.
 *
 * @param box   is a polygon representing a bounding box.
 * @param boxes is a vector of polygons representing boxes.
 *
 * @returns a vector of floats containing the ious of each box in `boxes` with
 * `box`.
 */
template <typename T>
[[nodiscard]] inline std::vector<T> iou(const polygon_t &box,
                                        const std::vector<polygon_t> &boxes) {
  std::vector<T> ious(boxes.size());

  // NOTE(tom): I have parallelized this, but this might only be worth it for
  //            larger sizes `boxes`. Should be perf tested.
  std::transform(
      std::execution::par_unseq, boxes.begin(), boxes.end(), ious.begin(),

      [box](const polygon_t &b) -> T {
        multi_polygon_t mpi;
        multi_polygon_t mpu;

        if (boost::geometry::intersects(box, b)) {
          boost::geometry::intersection(box, b, mpi);
          boost::geometry::union_(box, b, mpu);

          return boost::geometry::area(mpi) / boost::geometry::area(mpu);

        } else {
          return 0;
        }
      }

  );

  return ious;
}

} // namespace evaluation_utils

namespace cpp_utils {
/**
 * Returns the indices that sort a stl Container in ascending order by value.
 *
 * @tparam T is the type of the contents of the container that needs to be
 *           sorted, needs to be comparable.
 *
 * @param c             is the input container with the unsorted items.
 * @param descending    determines whether it is supposed to be sorted in
 *                      ascending or descending order.
 *                      Optional and defaults to false.
 *
 * @return              a `Container` with the indices sorted by value.
 */
template <template <typename...> class Container, typename T>
[[nodiscard]] Container<std::size_t> argsort(const Container<T> &c,
                                             bool descending = false);

} // namespace cpp_utils

#endif // !UTILS_HPP

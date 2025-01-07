#ifndef UTILS_HPP
#define UTILS_HPP

#include "boost/geometry/algorithms/detail/distance/interface.hpp"
#include "boost/geometry/core/coordinate_dimension.hpp"
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
#include <numeric>
#include <stdexcept>
#include <torch/serialize/tensor.h>
#include <vector>

namespace rt {

template <typename T> struct vec3 {
  T x, y, z;

  vec3<T>(const torch::Tensor &v)
      : x(v[0].item<T>()), y(v[1].item<T>()), z(v[2].item<T>()){};
  vec3<T>(T _x, T _y, T _z) : x(_x), y(_y), z(_z){};

  [[nodiscard]] torch::Tensor get_tensor() {
    return torch::tensor({this->x, this->y, this->z});
  }
  vec3<T> &operator/=(const T rhs) {
    if (std::is_arithmetic_v<T> && rhs == 0)
      throw std::invalid_argument("Cannot divide by 0!");

    this->x /= rhs;
    this->y /= rhs;
    this->z /= rhs;
    return *this;
  }
  friend vec3<T> operator*(const vec3 &lhs, const T rhs) {
    return vec3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
  }
  friend vec3<T> operator/(const vec3 &lhs, const T rhs) {
    if (std::is_arithmetic_v<T> && rhs == 0)
      throw std::invalid_argument("Cannot divide by 0!");

    return vec3<T>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
  }
  friend vec3<T> operator+(const vec3 &lhs, const vec3 &rhs) {
    return vec3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
  }

  friend bool operator==(const vec3 &lhs, const vec3 &rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
  }
  friend std::ostream &operator<<(std::ostream &outs, const vec3 &v) {
    return outs << "{" << v.x << ", " << v.y << ", " << v.z << "}";
  }
};

} // namespace rt

namespace math_utils {

constexpr float PI_DEG = 180.0;
constexpr float PI_RAD = static_cast<float>(M_PI);
constexpr float TWO_PI_RAD = 2.0f * PI_RAD;
constexpr double sqrt2 = 1.4142135623730951;

/**
 * Generates a rotation matrix around the 'z' axis (yaw) from the provided
 * angle.
 *
 * @param angle is the angle (in radians)
 *
 * @returns a 3x3 rotation matrix (in form of a torch::Tensor)
 */
[[nodiscard]] inline torch::Tensor rotate_yaw(const float angle) {

  float cos_angle = cos(angle);
  float sin_angle = sin(angle);

  auto rotation = torch::tensor({{cos_angle, 0.0f, sin_angle},
                                 {0.0f, 1.0f, 0.0f},
                                 {-sin_angle, 0.0f, cos_angle}});
  return rotation;
}

[[nodiscard]] constexpr inline float to_rad(const float angle) noexcept {
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
constexpr auto I32 = torch::kI32;
constexpr auto I64 = torch::kI64;

[[nodiscard]] torch::Tensor rotate_yaw_t(const torch::Tensor &points,
                                         const torch::Tensor &angle);

/**
 * Converts bounding boxes tensor with shape (N, 7) to a different tensor with
 * shape (N, 8, 3) where the boxes are represented using their 8 corners and
 * their coordinates in space.
 *
 *      7 -------- 4
 *     /|         /|
 *    6 -------- 5 .
 *    | |        | |
 *    . 3 -------- 0
 *    |/         |/
 *    2 -------- 1
 *
 *  @param boxes: is the input: (N, 7) with
 *                [x, y, z, dx, dy, dz, heading],
 *                and (x, y, z) as the box center
 *
 *  @returns: a new tensor (shape: (N, 8, 3)).
 *
 */
[[nodiscard]] torch::Tensor boxes_to_corners(const torch::Tensor &boxes);

} // namespace torch_utils

namespace evaluation_utils {

using point2d_t =
    boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian>;
using point3d_t =
    boost::geometry::model::point<float, 3, boost::geometry::cs::cartesian>;

template <typename point_t,
          typename = std::enable_if_t<std::is_same_v<point_t, point2d_t> ||
                                      std::is_same_v<point_t, point3d_t>>>
using polygon_t = boost::geometry::model::polygon<point_t, false>;

template <typename point_t,
          typename = std::enable_if_t<std::is_same_v<point_t, point2d_t> ||
                                      std::is_same_v<point_t, point3d_t>>>
using multi_polygon_t =
    boost::geometry::model::multi_polygon<polygon_t<point_t>>;

using polygon3d_t = polygon_t<point3d_t>;
using multi_polygon3d_t = multi_polygon_t<point3d_t>;

using polygon2d_t = polygon_t<point2d_t>;
using multi_polygon2d_t = multi_polygon_t<point2d_t>;

template <typename point_t>
[[nodiscard]] auto make_polygon(const torch::Tensor &box) {

  static_assert(std::is_same_v<point_t, point2d_t> ||
                std::is_same_v<point_t, point3d_t>);

  if constexpr (boost::geometry::dimension<point_t>::value == 2) {
    point2d_t p1{box[0][0].item<float>(), box[0][1].item<float>()};
    point2d_t p2{box[1][0].item<float>(), box[1][1].item<float>()};
    point2d_t p3{box[2][0].item<float>(), box[2][1].item<float>()};
    point2d_t p4{box[3][0].item<float>(), box[3][1].item<float>()};

    return polygon_t<point_t>{{p1, p2, p3, p4, p1}};
  } else if constexpr (boost::geometry::dimension<point_t>::value == 3) {

    // clang-format off
    point3d_t p1{box[0][0].item<float>(),
                 box[0][1].item<float>(),
                 box[0][2].item<float>()};

    point3d_t p2{box[1][0].item<float>(),
                 box[1][1].item<float>(),
                 box[1][2].item<float>()};

    point3d_t p3{box[2][0].item<float>(),
                 box[2][1].item<float>(),
                 box[2][2].item<float>()};

    point3d_t p4{box[3][0].item<float>(),
                 box[3][1].item<float>(),
                 box[3][2].item<float>()};

    point3d_t p5{box[4][0].item<float>(),
                 box[4][1].item<float>(),
                 box[4][2].item<float>()};

    point3d_t p6{box[5][0].item<float>(),
                 box[5][1].item<float>(),
                 box[5][2].item<float>()};

    point3d_t p7{box[6][0].item<float>(),
                 box[6][1].item<float>(),
                 box[6][2].item<float>()};

    point3d_t p8{box[7][0].item<float>(),
                 box[7][1].item<float>(),
                 box[7][2].item<float>()};
    // clang-format on

    return polygon_t<point_t>{{p1, p2, p3, p4, p5, p6, p7, p8, p1}};
  }
}

template <typename point_t>
[[nodiscard]] inline std::vector<polygon_t<point_t>>
convert_format(const torch::Tensor &boxes) {

  const auto corners = torch_utils::boxes_to_corners(boxes);

  std::vector<polygon_t<point_t>> ps;
  ps.reserve(static_cast<std::size_t>(corners.size(0)));

  for (tensor_size_t i = 0; i < corners.size(0); i++) {
    auto box = corners[i];

    auto p = make_polygon<point_t>(box);

    ps.emplace_back(p);
  }

  return ps;
}

/**
 * Computes intersection over union between (2D) `gt_box` and `boxes`.
 *
 * @param gt_box is a polygon representing a bounding box.
 * @param boxes  is a vector of polygons representing boxes.
 *
 * @returns a vector of floats containing the ious of each box in `boxes` with
 * `box`.
 */
template <typename T>
[[nodiscard]] inline std::vector<T>
iou_2d(const polygon2d_t &gt_box, const std::vector<polygon2d_t> &boxes) {
  std::vector<T> ious(boxes.size());

  std::transform(boxes.begin(), boxes.end(), ious.begin(),

                 [gt_box](const polygon2d_t &box) -> T {
                   if (boost::geometry::intersects(gt_box, box)) {
                     multi_polygon2d_t mpu;
                     multi_polygon2d_t mpi;

                     boost::geometry::intersection(gt_box, box, mpi);
                     boost::geometry::union_(gt_box, box, mpu);

                     return boost::geometry::area(mpi) /
                            boost::geometry::area(mpu);
                   }

                   return 0;
                 }

  );

  return ious;
}

/**
 * Computes intersection over union between (a 3D) `gt_box` and `boxes`.
 *
 * @param gt_box is a polygon representing a bounding box.
 * @param boxes  is a vector of polygons representing boxes.
 *
 * @returns a vector of floats containing the ious of each box in `boxes` with
 * `box`.
 */
template <typename T>
[[nodiscard]] inline std::vector<T>
iou_3d(const polygon3d_t &gt_box, const std::vector<polygon3d_t> &boxes) {
  std::vector<T> ious;
  ious.resize(boxes.size());

  const auto &gt_box_outer = gt_box.outer();
  polygon2d_t gt_2d{{
      {gt_box_outer[0].get<0>(), gt_box.outer()[0].get<2>()},
      {gt_box_outer[1].get<0>(), gt_box.outer()[1].get<2>()},
      {gt_box_outer[2].get<0>(), gt_box.outer()[2].get<2>()},
      {gt_box_outer[3].get<0>(), gt_box.outer()[3].get<2>()},
  }};

  boost::geometry::correct(gt_2d);

  const T gt_box_height =
      boost::geometry::distance(gt_box_outer[1], gt_box_outer[5]);

  std::transform(
      boxes.begin(), boxes.end(), ious.begin(),

      [gt_2d, gt_box_outer, gt_box_height](const polygon3d_t &box) -> T {
        const auto &box_outer = box.outer();

        // get base
        polygon2d_t box_2d{{
            {box_outer[0].get<0>(), box_outer[0].get<2>()},
            {box_outer[1].get<0>(), box_outer[1].get<2>()},
            {box_outer[2].get<0>(), box_outer[2].get<2>()},
            {box_outer[3].get<0>(), box_outer[3].get<2>()},
        }};

        boost::geometry::correct(box_2d);

        if (boost::geometry::intersects(gt_2d, box_2d)) {
          T y_high = fmin(box_outer[5].get<1>(), gt_box_outer[5].get<1>());
          T y_low = fmax(box_outer[1].get<1>(), gt_box_outer[1].get<1>());

          // intersection area > 0
          if (y_high > y_low) {
            multi_polygon2d_t mpi;

            boost::geometry::intersection(gt_2d, box_2d, mpi);

            const T intersection_area =
                boost::geometry::area(mpi) * (y_high - y_low);

            const T box_height =
                boost::geometry::distance(box_outer[1], box_outer[5]);

            const T boxes_union = boost::geometry::area(box_2d) * box_height +
                                  boost::geometry::area(gt_2d) * gt_box_height -
                                  intersection_area;

            return intersection_area / boxes_union;
          }
        }

        return 0;
      }

  );

  return ious;
}

template <typename T, typename point_t>
[[nodiscard]] inline std::vector<T>
iou(const polygon_t<point_t> &gt_box, std::vector<polygon_t<point_t>> &boxes) {

  if constexpr (boost::geometry::dimension<point_t>::value == 2) {
    return iou_2d<float>(gt_box, boxes);
  }

  if constexpr (boost::geometry::dimension<point_t>::value == 3) {
    return iou_3d<float>(gt_box, boxes);
  }
}

} // namespace evaluation_utils

namespace cpp_utils {

template <typename T> struct range {
  T min, max;
};

template <typename T> struct distribution_ranges {
  range<T> x_range, y_range, z_range, uniform_range;

  constexpr distribution_ranges<T>(range<T> x_range, range<T> y_range,
                                   range<T> z_range, range<T> uniform_range)
      : x_range(x_range), y_range(y_range), z_range(z_range),
        uniform_range(uniform_range){};
  constexpr distribution_ranges<T>(range<T> x_range, range<T> y_range,
                                   range<T> z_range)
      : x_range(x_range), y_range(y_range), z_range(z_range),
        uniform_range(range<T>{0, 0}){};
};

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
 * @returns             a `Container` with the indices sorted by value.
 */
template <template <typename...> class Container, typename T>
[[nodiscard]] Container<std::size_t> argsort(const Container<T> &c,
                                             const bool descending = false) {

  Container<size_t> idx(c.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (descending) {
    std::stable_sort(idx.begin(), idx.end(),
                     [&c](size_t i1, size_t i2) { return c[i1] > c[i2]; });

  } else {
    std::stable_sort(idx.begin(), idx.end(),
                     [&c](size_t i1, size_t i2) { return c[i1] < c[i2]; });
  }

  return idx;
}

} // namespace cpp_utils

#endif // !UTILS_HPP

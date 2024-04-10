#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <torch/serialize/tensor.h>

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

} // namespace math_utils

namespace torch_utils {
constexpr auto F32 = torch::kF32;
constexpr auto F64 = torch::kF64;
} // namespace torch_utils

#endif // !UTILS_HPP

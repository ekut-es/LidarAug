
#include "../include/utils.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/cos.h>

using Slice = torch::indexing::Slice;
constexpr auto None = torch::indexing::None;

namespace torch_utils {

[[nodiscard]] torch::Tensor boxes_to_corners(const torch::Tensor &boxes) {

  const auto t = torch::tensor({
                     {1, -1, -1},
                     {1, 1, -1},
                     {-1, 1, -1},
                     {-1, -1, -1},
                     {1, -1, 1},
                     {1, 1, 1},
                     {-1, 1, 1},
                     {-1, -1, 1},
                 }) /
                 2;

  auto corners = boxes.index({Slice(), None, Slice(3, 6)}).repeat({1, 8, 1}) *
                 t.index({None, Slice(), Slice()});

  corners = rotate_yaw_t(corners.view({-1, 8, 3}), boxes.index({Slice(), 6}))
                .view({-1, 8, 3});

  corners += boxes.index({Slice(), None, Slice(0, 3)});
  return corners;
}

[[nodiscard]] torch::Tensor rotate_yaw_t(const torch::Tensor &points,
                                         const torch::Tensor &angle) {

  auto cos = torch::cos(angle);
  auto sin = torch::sin(angle);

  auto zeros = angle.new_zeros(points.size(0));
  auto ones = angle.new_ones(points.size(0));

  const auto rot_matrix =
      torch::stack({cos, sin, zeros, -sin, cos, zeros, zeros, zeros, ones}, 1)
          .view({-1, 3, 3});

  auto points_rot =
      torch::matmul(points.index({Slice(), Slice(), Slice(0, 3)}), rot_matrix);
  points_rot = torch::cat(
      {points_rot, points.index({Slice(), Slice(), Slice(3, None)})}, -1);

  return points_rot;
}

} // namespace torch_utils



#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <cstdint>
#include <torch/serialize/tensor.h>

// return type of `at::Tensor::size`
using tensor_size_t = std::int64_t;

using dimensions = struct {
  tensor_size_t batch_size, num_items, num_features;
};

/**
 * Changes the representation of a sparse tensor from a flat 2D tensor (N, F),
 * where F is the number of features to a 3D tensor (B, n, f), where B is the
 * number of batches, n is the number of tensors in each batch and f is the
 * number of features (equal to F-1).
 * 0s are used for padding.
 *
 * @param input     is the input tensor.
 * @param batch_idx is the index of the batch index.
 *
 * @returns a new tensor with 0 padding.
 */
[[nodiscard]] torch::Tensor
change_sparse_representation(const torch::Tensor &input,
                             tensor_size_t batch_idx);

#endif // !TENSOR_HPP



#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <cstdint>

// return type of `at::Tensor::size`
using tensor_size_t = std::int64_t;

typedef struct {
  tensor_size_t batch_size, num_items, num_features;
} dimensions;

#endif // !TENSOR_HPP

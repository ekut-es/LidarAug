
#include "../include/tensor.hpp"
#include <ATen/core/TensorAccessor.h>
#include <ATen/ops/stack.h>
#include <torch/torch.h>
#include <vector>

using size_t = std::size_t;

[[nodiscard]] torch::Tensor
change_sparse_representation(const torch::Tensor &input,
                             const tensor_size_t batch_idx) {

  const auto num_tensors = input.size(0);
  const auto num_features = input.size(1);

  auto determine_shape =
      [num_tensors](const torch::TensorAccessor<tensor_size_t, 2> &in,
                    const tensor_size_t batch_idx_) {
        tensor_size_t num_batches = 0;
        size_t max_count = 0;
        size_t current_count = 0;

        for (tensor_size_t i = 0; i < num_tensors; i++) {

          if (const auto batch_num = in[i][batch_idx_];
              batch_num == num_batches) {
            current_count++;
          }

          else if (batch_num > num_batches) {
            num_batches = batch_num;

            if (current_count > max_count) {
              max_count = current_count;
            }
          } else {
            // NOTE(tom): reaching this means that the tensor is not ordered,
            //            which is unexpected and not handled by this function.
            assert(false);
          }
        }

        // index to count conversion
        num_batches++;

        return std::make_pair(num_batches, max_count);
      };

  auto gather_tensors = [num_tensors, num_features](
                            const torch::Tensor &in,
                            const tensor_size_t batch_idx_,
                            const size_t num_batches, tensor_size_t max_count) {
    std::vector<torch::Tensor> batch;
    batch.reserve(num_batches);

    auto tensor = torch::zeros({max_count, num_features - 1});
    tensor_size_t tensor_idx = 0;

    for (tensor_size_t i = 0; i < num_tensors; i++) {
      const auto batch_num =
          static_cast<size_t>(in.accessor<tensor_size_t, 2>()[i][batch_idx_]);

      std::cout << "batch_num: " << batch_num << "\n";

      if (batch_num > batch.size()) {
        batch.emplace_back(tensor.clone(torch::get_contiguous_memory_format()));
        tensor_idx = 0;
        std::ignore = tensor.zero_();
      }

      tensor.index_put_({tensor_idx++}, in[i]);
    }

    batch.emplace_back(tensor.clone(torch::get_contiguous_memory_format()));

    return torch::stack(batch);
  };

  auto [num_batches, max_count] =
      determine_shape(input.accessor<tensor_size_t, 2>(), batch_idx);

  std::cout << "B: " << num_batches << "\n";
  std::cout << "n: " << max_count << std::endl;

  return gather_tensors(input.slice(1, 0, num_features - 1), batch_idx,
                        static_cast<size_t>(num_batches),
                        static_cast<tensor_size_t>(max_count));
}


#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

using result_dict =
    std::map<std::uint8_t, std::map<std::string, std::vector<float>>>;

/**
 * Calculates the false and true positive numbers of the current frames.
 *
 * @param detection_boxes  are the detection bounding box.
 *                         The expected tensor shape is (N, 7).
 * @param detection_score  is the confidence score for each predicted bounding
 *                         box.
 * @param ground_truth_box is the ground truth bounding box.
 * @param iou_threshold    is the minimum 'intersection over union' threshold.
 * @param results          is an unordered map containing the false- & true
 *                         positive numbers as well as the ground truth.
 */
template <typename point_t>
void calculate_false_and_true_positive(const torch::Tensor &detection_boxes,
                                       const torch::Tensor &detection_score,
                                       const torch::Tensor &ground_truth_box,
                                       float iou_threshold,
                                       result_dict &results);

template <typename T>
[[nodiscard]] T calculate_average_precision(
    float iou_threshold, bool global_sort_detections,
    const std::map<std::uint8_t, std::map<std::string, std::vector<T>>>
        &results);

/**
 * Calculates the Visual Object Classes (VOC) Challenge 2010 average precision.
 *
 * @tparam T        is the (numeric) type of the average precision.
 * @param recall    are the recall values used to calculte the mean recall.
 * @param precision are the precision values used to calculate mean precision.
 * @returns          the average precision.
 */
template <typename T>
[[nodiscard]] inline T
calculate_voc_average_precision(const std::vector<T> &recall,
                                const std::vector<T> &precision) {
  std::vector<T> mean_recall(recall.size() + 2);
  mean_recall[0] = 0;
  mean_recall[mean_recall.size() - 1] = 1;

  std::transform(recall.begin(), recall.end(), mean_recall.begin() + 1,
                 [](const T val) { return val; });

  std::vector<T> mean_precision(precision.size() + 2);
  mean_precision[0] = 0;
  mean_precision[mean_precision.size() - 1] = 0;

  std::transform(precision.begin(), precision.end(), mean_precision.begin() + 1,
                 [](const T val) { return val; });

  for (std::int64_t i = mean_precision.size() - 2; i >= 0; --i) {
    mean_precision[i] = std::max(mean_precision[i], mean_precision[i + 1]);
  }

  std::vector<std::size_t> indices;
  indices.reserve(mean_recall.size());

  for (std::size_t i = 1; i < mean_recall.size(); i++) {
    if (mean_recall[i] != mean_recall[i - 1]) {
      indices.emplace_back(i);
    }
  }

  T average_precision = 0;

  for (auto i : indices) {
    average_precision +=
        (mean_recall[i] - mean_recall[i - 1]) * mean_precision[i];
  }

  return average_precision;
}

/**
 * Calculates the average precision for different iou thresholds and writes the
 * result to a yaml file.
 *
 * @tparam T is the type of the average precision.
 *
 * @param results                a map with the results.
 * @param global_sort_detections ?
 */
std::array<float, 3> evaluate_results(const result_dict &results,
                                      bool global_sort_detections);

#endif // !EVALUATION_HPP

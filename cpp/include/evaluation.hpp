
#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <map>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

template <typename T> struct result_dict {
  typedef std::map<float, std::unordered_map<std::string, std::vector<T>>> type;
};

/**
 * Calculates the false and true positive numbers of the current frames.
 *
 * @param detection_boxes  are the detection bounding box.
 *                         Their shape is either (N, 8, 3) or (N, 4, 2).
 * @param detection_score  is the confidence score for each predicted bounding
 * box.
 * @param ground_truth_box is the ground truth bounding box.
 * @param iou_threshold    is the minimum 'intersection over union' threshold.
 * @param results          is an unordered map containing the false- & true
 *                         positive numbers as well as the ground truth.
 */
void calculate_false_and_true_positive(
    const torch::Tensor &detection_boxes, const torch::Tensor &detection_score,
    const torch::Tensor &ground_truth_box, float iou_threshold,
    typename result_dict<float>::type results);
#endif // !EVALUATION_HPP

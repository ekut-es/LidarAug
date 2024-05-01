#include "../include/evaluation.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"

void calculate_false_and_true_positive(
    const torch::Tensor &detection_boxes, torch::Tensor detection_score,
    const torch::Tensor &ground_truth_box, float iou_threshold,
    typename result_dict<float>::type results) {

  assert(detection_score.is_contiguous());

  auto data = detection_score.data_ptr<float>();

  std::vector<float> l_detection_score(
      data, data + static_cast<std::size_t>(detection_score.size(0)));

  detection_score.sort(-1, true);

  std::vector<float> true_positive;
  std::vector<float> false_positive;

  true_positive.reserve(l_detection_score.size());
  false_positive.reserve(l_detection_score.size());

  auto ground_truth = ground_truth_box.size(0);

  auto score_order_descend = cpp_utils::argsort(l_detection_score, true);

  auto detection_polygon_list =
      evaluation_utils::convert_format(detection_boxes);
  auto ground_truth_polygon_list =
      evaluation_utils::convert_format(ground_truth_box);

  // match prediction and ground truth bounding box
  for (const auto idx : score_order_descend) {
    const auto detection_polygon = detection_polygon_list[idx];
    auto ious = evaluation_utils::iou<float>(detection_polygon,
                                             ground_truth_polygon_list);

    if (ground_truth_polygon_list.empty() ||
        // NOTE(tom): I have parallelized this, but this might only be worth it
        //            for larger sizes `boxes`. Should be perf tested.
        *std::max_element(std::execution::par_unseq, ious.begin(), ious.end()) <
            iou_threshold) {

      false_positive.push_back(1);
      true_positive.push_back(0);

    } else {
      false_positive.push_back(0);
      true_positive.push_back(1);

      auto gt_indices = torch::argmax(torch::from_blob(
          ious.data(), static_cast<tensor_size_t>(ious.size())));

      for (tensor_size_t j = 0; j < gt_indices.size(0); j++) {
        ground_truth_polygon_list.erase(ground_truth_polygon_list.begin() +
                                        gt_indices[j].item<tensor_size_t>());
      }
    }
  }

  auto iout = results[iou_threshold];
  auto sc = iout["SCORE"];
  auto fp = iout["FALSE_POSITIVE"];
  auto tp = iout["TRUE_POSITIVE"];
  auto gt = iout["GROUND_TRUTH"];

  sc.insert(sc.end(), l_detection_score.begin(), l_detection_score.end());
  fp.insert(fp.end(), false_positive.begin(), false_positive.end());
  tp.insert(tp.end(), true_positive.begin(), true_positive.end());
  gt.emplace_back(ground_truth);
}

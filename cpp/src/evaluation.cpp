#include "../include/evaluation.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <cstdio>

template <typename T>
T calculate_average_precision(
    const float iou_threshold, const bool global_sort_detections,
    const std::map<std::uint8_t, std::map<std::string, std::vector<T>>>
        &results) {

  auto iou = results.at(static_cast<std::uint8_t>(iou_threshold * 10));

  auto false_positive = iou["fp"];
  auto true_positive = iou["tp"];

  if (global_sort_detections) {

    auto score = iou["score"];

    assert(false_positive.size() == true_positive.size() &&
           true_positive.size() == score.size());

    std::sort(false_positive.begin(), false_positive.end());
    std::sort(true_positive.begin(), true_positive.end());

  } else {
    assert(false_positive.size() == true_positive.size());
  }

  // NOTE(tom): This is a single value but it has to be in vector form because
  //            of type constraints
  auto ground_truth = iou["gt"][0];

  auto sum = 0;

  for (std::size_t i = 0; i < false_positive.size(); i++) {
    auto val = false_positive[i];
    false_positive[i] += sum;
    sum += val;
  }

  sum = 0;

  for (std::size_t i = 0; i < true_positive.size(); i++) {
    auto val = true_positive[i];
    true_positive[i] += sum;
    sum += val;
  }

  std::vector<T> recall(true_positive.size());
  std::transform(
      true_positive.begin(), true_positive.end(), recall.begin(),
      [ground_truth](const T tp_val) { return tp_val / ground_truth; });

  std::vector<T> precision;
  precision.reserve(true_positive.size());

  for (std::size_t i = 0; i < precision.capacity(); i++) {
    precision.emplace_back(static_cast<T>(true_positive[i]) /
                           (false_positive[i] + true_positive[i]));
  }

  return calculate_voc_average_precision<T>(recall, precision);
}

template <typename point_t>
void calculate_false_and_true_positive(const torch::Tensor &detection_boxes,
                                       const torch::Tensor &detection_score,
                                       const torch::Tensor &ground_truth_box,
                                       float iou_threshold,
                                       result_dict &results) {

  assert(detection_score.is_contiguous());

  const auto *const data = detection_score.const_data_ptr<float>();

  const std::vector<float> l_detection_score(
      data, data + static_cast<std::size_t>(detection_score.size(0)));

  std::vector<float> true_positive;
  std::vector<float> false_positive;

  true_positive.reserve(l_detection_score.size());
  false_positive.reserve(l_detection_score.size());

  auto ground_truth = ground_truth_box.size(0);

  const auto score_order_descend = cpp_utils::argsort(l_detection_score, true);

  const auto detection_polygon_list =
      evaluation_utils::convert_format<point_t>(detection_boxes);
  auto ground_truth_polygon_list =
      evaluation_utils::convert_format<point_t>(ground_truth_box);

  // match prediction and ground truth bounding box
  for (const auto idx : score_order_descend) {
    const auto detection_polygon = detection_polygon_list[idx];

    std::vector<float> ious = evaluation_utils::iou<float, point_t>(
        detection_polygon, ground_truth_polygon_list);

    // NOTE(tom): This depends on the left condition being evaluated first!
    if (ground_truth_polygon_list.empty() ||
        *std::max_element(ious.begin(), ious.end()) < iou_threshold) {

      false_positive.emplace_back(1);
      true_positive.emplace_back(0);
    } else {
      false_positive.emplace_back(0);
      true_positive.emplace_back(1);

      const auto gt_index =
          torch::argmax(
              torch::from_blob(ious.data(),
                               static_cast<tensor_size_t>(ious.size())))
              .item<tensor_size_t>();

      ground_truth_polygon_list.erase(ground_truth_polygon_list.begin() +
                                      gt_index);
    }
  }

  auto insert_list = [iou_threshold, &results](const std::string &key,
                                               const std::vector<float> &list) {
    // NOTE(tom): this is necessary for some reason because pybind
    results[static_cast<std::uint8_t>(iou_threshold * 10)][key].insert(
        results[static_cast<std::uint8_t>(iou_threshold * 10)][key].end(),
        list.begin(), list.end());
  };

  insert_list("score", l_detection_score);
  insert_list("fp", false_positive);
  insert_list("tp", true_positive);

  // set ground truth
  if (results[static_cast<std::uint8_t>(iou_threshold * 10)]["gt"].empty()) {
    results[static_cast<std::uint8_t>(iou_threshold * 10)]["gt"].emplace_back(
        ground_truth);
  } else {

    results[static_cast<std::uint8_t>(iou_threshold * 10)]["gt"][0] +=
        static_cast<float>(ground_truth);
  }
}

std::array<float, 3> evaluate_results(const result_dict &results,
                                      bool global_sort_detections) {

  constexpr std::array<float, 3> iou_thresholds{.3, .5, .7};
  std::array<float, 3> aps;

  std::transform(iou_thresholds.begin(), iou_thresholds.end(), aps.begin(),
                 [global_sort_detections, results](auto threshold) {
                   auto ap = calculate_average_precision(
                       threshold, global_sort_detections, results);

                   std::printf("ap_%f: %f\n", threshold, ap);

                   return ap;
                 });

  return aps;
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "../include/evaluation_bindings.hpp"
#endif

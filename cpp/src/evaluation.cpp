#include "../include/evaluation.hpp"
#include "../include/tensor.hpp"
#include "../include/utils.hpp"
#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

template <typename T>
T calculate_average_precision(float iou_threshold, bool global_sort_detections,
                              typename result_dict<T>::type results) {

  auto iou = results[iou_threshold];

  auto false_positive = iou["FALSE_POSITIVE"];
  auto true_positive = iou["TRUE_POSITIVE"];
  if (global_sort_detections) {

    auto score = iou["SCORE"];

    assert(false_positive.size() == true_positive.size() &&
           true_positive.size() == score.size());

    auto sorted_index = cpp_utils::argsort(score, true);

    false_positive = false_positive[sorted_index];
    true_positive = true_positive[sorted_index];
  } else {
    assert(false_positive.size() == true_positive.size());
  }

  auto ground_truth = iou["GROUND_TRUTH"];

  auto sum = 0;

  for (std::size_t i = 0; i < false_positive.size(); i++) {
    sum += false_positive[i];
    false_positive[i] += sum;
  }

  sum = 0;

  for (std::size_t i = 0; i < true_positive.size(); i++) {
    sum += true_positive[i];
    true_positive[i] += sum;
  }

  auto recall = true_positive;

  for (std::size_t i = 0; i < true_positive.size(); i++) {
    recall[i] = static_cast<float>(true_positive[i]) / ground_truth;
  }

  auto precision = true_positive;

  for (std::size_t i = 0; i < true_positive.size(); i++) {
    precision[i] = static_cast<float>(true_positive[i]) /
                   (false_positive[i] + true_positive[i]);
  }

  return calculate_voc_average_precision<T>(recall, precision);
}

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

template <typename T>
void evaluate_results(typename result_dict<T>::type results, std::string dir,
                      bool global_sort_detections) {

  std::array<float, 3> iou_thresholds{.3, .5, .7};

  YAML::Emitter o;
  o << YAML::BeginMap;

  std::ranges::for_each(iou_thresholds, [&o, global_sort_detections,
                                         results](auto threshold) {
    auto ap =
        calculate_average_precision(threshold, global_sort_detections, results);

    o << YAML::Key << std::format("ap_%f", threshold);
    o << YAML::Value << ap;
  });

  o << YAML::EndMap;

  std::cout << o.c_str();

  // writing yaml to file
  if (!dir.ends_with('/'))
    dir.append("/");

  dir.append(global_sort_detections ? "eval_global_sort.yaml" : "eval.yaml");

  std::ofstream fout(dir);
  fout << o.c_str();
}

#ifdef BUILD_MODULE
#undef TEST_RNG
#include "../include/evaluation_bindings.hpp"
#endif

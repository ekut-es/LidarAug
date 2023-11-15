
#include "../include/transformations.hpp"
#include <gtest/gtest.h>

TEST(TranslationTest, BasicAssertions) {
  auto tensor = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
  auto translation = torch::tensor({1.0, 2.0, 3.0});

  translate(tensor, translation);

  auto expected = torch::tensor({{{2.0, 4.0, 6.0}, {5.0, 7.0, 9.0}}});

  ASSERT_TRUE(tensor.equal(expected));
}

TEST(ScalingTest, BasicAssertions) {
  auto tensor = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
  auto labels = torch::tensor(
      {{1.0, 1.0, 1.0, 2.0, 3.0, 2.5}, {2.0, 2.0, 2.0, 1.0, 1.0, 0.5}});

  auto scaling_factor = 2.0;

  scale_points(tensor, scaling_factor);
  scale_labels(labels, scaling_factor);

  auto expected_points = torch::tensor({{{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}}});
  auto expected_labels = torch::tensor(
      {{2.0, 2.0, 2.0, 4.0, 6.0, 5.0}, {4.0, 4.0, 4.0, 2.0, 2.0, 1.0}});

  ASSERT_TRUE(tensor.equal(expected_points));
  ASSERT_TRUE(labels.equal(expected_labels));
}

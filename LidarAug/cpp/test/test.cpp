
#include "../include/transformations.hpp"
#include <gtest/gtest.h>
#include <torch/types.h>

// NOLINTBEGIN

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
      {{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5}, {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}});

  auto scaling_factor = 2.0;

  scale_points(tensor, scaling_factor);
  scale_labels(labels, scaling_factor);

  auto expected_points = torch::tensor({{{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}}});
  auto expected_labels = torch::tensor(
      {{{2.0, 2.0, 2.0, 4.0, 6.0, 5.0}, {4.0, 4.0, 4.0, 2.0, 2.0, 1.0, M_PI}}});

  ASSERT_TRUE(tensor.equal(expected_points));
  ASSERT_TRUE(labels.equal(expected_labels));
}

TEST(FlipTest, BasicAssertions) {
  {
    auto points = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
    auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.5},
                                  {2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5}}},
                                torch::kFloat16);

    static constexpr auto probability = 100;

    auto expected_points =
        torch::tensor({{{1.0, -2.0, 3.0}, {4.0, -5.0, 6.0}}});
    auto expected_labels =
        torch::tensor({{{1.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.641592653589793},
                        {2.0, -2.0, 2.0, 1.0, 1.0, 1.5, 3.641592653589793}}},
                      {torch::kFloat16});

    flip_random(points, labels, probability);

    EXPECT_TRUE(points.equal(expected_points))
        << "Expected: \n"
        << expected_points << "\nActual: \n"
        << points;

    EXPECT_TRUE(labels.equal(expected_labels))
        << "Expected: \n"
        << expected_labels << "\nActual: \n"
        << labels;
  }
  {
    auto points = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
    auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.5},
                                  {2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5}}});

    static constexpr auto probability = 0;

    flip_random(points, labels, probability);

    EXPECT_TRUE(points.equal(points)) << "Points unexpectidly changed";
    EXPECT_TRUE(labels.equal(labels)) << "Labels unexpectidly changed";
  }
}

// NOLINTEND


#include "../include/stats.hpp"
#include "../include/transformations.hpp"
#include "../include/utils.hpp"
#include <gtest/gtest.h>
#include <torch/types.h>

// NOLINTBEGIN

TEST(TranslationTest, BasicAssertions) {
  auto tensor = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
  auto translation = torch::tensor({1.0, 2.0, 3.0});

  translate(tensor, translation);

  auto expected = torch::tensor({{{2.0, 4.0, 6.0}, {5.0, 7.0, 9.0}}});

  EXPECT_TRUE(tensor.equal(expected));
}

TEST(ScalingTest, BasicAssertions) {
  auto tensor = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}});

  auto scaling_factor = 2.0;

  scale_points(tensor, scaling_factor);
  scale_labels(labels, scaling_factor);

  auto expected_points = torch::tensor({{{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}}});
  auto expected_labels =
      torch::tensor({{{2.0, 2.0, 2.0, 4.0, 6.0, 5.0, M_PI},
                      {4.0, 4.0, 4.0, 2.0, 2.0, 1.0, M_PI}}});

  EXPECT_TRUE(tensor.equal(expected_points));
  EXPECT_TRUE(labels.equal(expected_labels));
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

    auto expected_points = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
    auto expected_labels =
        torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.5},
                        {2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5}}});

    flip_random(points, labels, probability);

    EXPECT_TRUE(points.equal(expected_points)) << "Points unexpectidly changed";
    EXPECT_TRUE(labels.equal(expected_labels)) << "Labels unexpectidly changed";
  }
}

TEST(AngleConversionTest, BasicAssertions) {
  constexpr static float three_sixty_deg = 360;
  constexpr static float zero_deg = 0;
  constexpr static float one_eighty_deg = 180;
  constexpr static float ninety_deg = 90;

  EXPECT_EQ(to_rad(three_sixty_deg), 2 * static_cast<float>(M_PI));
  EXPECT_EQ(to_rad(zero_deg), 0);
  EXPECT_EQ(to_rad(one_eighty_deg), static_cast<float>(M_PI));
  EXPECT_EQ(to_rad(-one_eighty_deg), static_cast<float>(-M_PI));
  EXPECT_EQ(to_rad(ninety_deg), static_cast<float>(M_PI / 2));
  EXPECT_EQ(to_rad(-ninety_deg), static_cast<float>((-M_PI) / 2));
}

TEST(DrawUniformValuesTest, BasicAssertions) {
  constexpr static auto size = 10;
  constexpr static auto num_values = 3;

  auto values = draw_unique_uniform_values<int>(size, num_values);

  EXPECT_EQ(values.size(), num_values);

  for (auto val : values) {
    EXPECT_LT(val, size);
    EXPECT_GE(val, 0);
  }
}

TEST(ThinOutTest, BasicAssertions) {
  auto points = torch::rand({2, 10, 4});
  dimensions dims_original = {points.size(0), points.size(1), points.size(2)};
  auto new_points = thin_out(points, 1);
  dimensions dims_edited = {new_points.size(0), new_points.size(1),
                            new_points.size(2)};

  EXPECT_EQ(dims_edited.batch_size, dims_original.batch_size);
  EXPECT_LT(dims_edited.num_items, dims_original.num_items)
      << "Expected the amounts to have reduced...\noriginal:\n"
      << points << "\nnew:\n"
      << new_points;
  EXPECT_EQ(dims_edited.num_features, dims_original.num_features);
}

// doing tests with controlled random number generation (no random seed)
#ifdef TEST_RNG

TEST(DrawValuesTest, BasicAssertions) {
  std::uniform_int_distribution<int> ud(0, 100);
  std::normal_distribution<float> nd(0, 5);
  {
    auto ud_result = std::get<0>(draw_values<int>(ud, 10));
    std::vector<int> ud_expected_values{70, 72, 28, 43, 22, 69, 55, 72, 72, 49};

    auto nd_result = std::get<0>(draw_values<float>(nd, 10));
    std::vector<float> nd_expected_values{
        5.42903471,  5.00874138, -2.83043623, -8.46256065,  3.64878798,
        -5.22126913, 8.69870472, 2.03683066,  -0.366712242, 9.06219864};

    EXPECT_EQ(ud_result, ud_expected_values)
        << "Vectors are not equal: expected\n"
        << ud_expected_values << "\nactual:\n"
        << ud_result;

    EXPECT_EQ(nd_result, nd_expected_values)
        << "Vectors are not equal: expected\n"
        << nd_expected_values << "\nactual:\n"
        << nd_result;
  }

  {
    auto ud_result = std::get<1>(draw_values<int>(ud));
    int expected_value = 70;

    auto nd_result = std::get<1>(draw_values<float>(nd));
    float nd_expected_value = 5.42903471;

    EXPECT_EQ(ud_result, expected_value) << "Values are not equal: expected\n"
                                         << expected_value << "\nactual:\n"
                                         << ud_result;

    EXPECT_EQ(nd_result, nd_expected_value)
        << "Values are not equal: expected\n"
        << nd_expected_value << "\nactual:\n"
        << nd_result;
  }
  {

    auto ud_result = std::get<0>(draw_values<int>(ud, {}, true));
    std::vector<int> ud_expected_value = {70};

    // NOTE(tom): for some reason this results in the second value from the
    // sequence, not the first
    auto nd_result = std::get<0>(draw_values<float>(nd, {}, true));
    std::vector<float> nd_expected_value{5.42903471};

    EXPECT_EQ(ud_result, ud_expected_value)
        << "Vectors are not equal: expected\n"
        << ud_expected_value << "\nactual:\n"
        << ud_result;

    // NOTE(tom): as a result of the note above; this will fail
    EXPECT_EQ(nd_result, nd_expected_value)
        << "Vectors are not equal: expected\n"
        << nd_expected_value << "\nactual:\n"
        << nd_result;
  }
}

#endif

// NOLINTEND


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

TEST(RotationTest, BasicAssertions) {

  {

    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}},
                       {{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                      torch::kF32);

    constexpr float ROT_ANGLE = 180.0f;
    rotate_deg(points, ROT_ANGLE);
    // rotate_random(points, labels, 50);

    auto points_coordinates =
        torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                       {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}},
                      torch::kF32);
    constexpr float ANGLE = ROT_ANGLE * (M_PI / PI_DEG);
    const auto rotation_vector = rotate_yaw(ANGLE);

    const auto v11 = torch::matmul(points_coordinates[0][0], rotation_vector);
    const auto v12 = torch::matmul(points_coordinates[0][1], rotation_vector);
    const auto v21 = torch::matmul(points_coordinates[1][0], rotation_vector);
    const auto v22 = torch::matmul(points_coordinates[1][1], rotation_vector);

    // The vectors need to contiguous for pointer pointer access
    ASSERT_TRUE(v11.is_contiguous());
    ASSERT_TRUE(v12.is_contiguous());
    ASSERT_TRUE(v21.is_contiguous());
    ASSERT_TRUE(v22.is_contiguous());

    const float *const vec11 = v11.const_data_ptr<float>();
    const float *const vec12 = v12.const_data_ptr<float>();
    const float *const vec21 = v21.const_data_ptr<float>();
    const float *const vec22 = v22.const_data_ptr<float>();

    const auto expected_points =
        torch::tensor({{{vec11[0], vec11[1], vec11[2], 10.0f},
                        {vec12[0], vec12[1], vec12[2], -10.0f}},
                       {{vec21[0], vec21[1], vec21[2], 10.0f},
                        {vec22[0], vec22[1], vec22[2], -10.0f}}},
                      torch::kF32);

    EXPECT_TRUE(points.allclose(expected_points))
        << "`points` not equal to `expected_points`:\npoints:" << points
        << "\nexpected_points:\n"
        << expected_points;
  }
  {

    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}},
                       {{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                      torch::kF32);

    constexpr float ANGLE = M_PI;
    rotate_rad(points, ANGLE);

    auto points_coordinates =
        torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                       {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}},
                      torch::kF32);

    const auto rotation_vector = rotate_yaw(ANGLE);

    const auto v11 = torch::matmul(points_coordinates[0][0], rotation_vector);
    const auto v12 = torch::matmul(points_coordinates[0][1], rotation_vector);
    const auto v21 = torch::matmul(points_coordinates[1][0], rotation_vector);
    const auto v22 = torch::matmul(points_coordinates[1][1], rotation_vector);

    // The vectors need to contiguous for pointer pointer access
    ASSERT_TRUE(v11.is_contiguous());
    ASSERT_TRUE(v12.is_contiguous());
    ASSERT_TRUE(v21.is_contiguous());
    ASSERT_TRUE(v22.is_contiguous());

    const float *const vec11 = v11.const_data_ptr<float>();
    const float *const vec12 = v12.const_data_ptr<float>();
    const float *const vec21 = v21.const_data_ptr<float>();
    const float *const vec22 = v22.const_data_ptr<float>();

    const auto expected_points =
        torch::tensor({{{vec11[0], vec11[1], vec11[2], 10.0f},
                        {vec12[0], vec12[1], vec12[2], -10.0f}},
                       {{vec21[0], vec21[1], vec21[2], 10.0f},
                        {vec22[0], vec22[1], vec22[2], -10.0f}}},
                      torch::kF32);

    EXPECT_TRUE(points.allclose(expected_points))
        << "`points` not equal to `expected_points`:\npoints:" << points
        << "\nexpected_points:\n"
        << expected_points;
  }
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

TEST(RandomNoiseTest, BasicAssertions) {
  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 10.9}, {4.0, 5.0, 6.0, -10.0}}});
  const auto points_original =
      torch::tensor({{{1.0, 2.0, 3.0, 10.9}, {4.0, 5.0, 6.0, -10.0}}});

  constexpr static float sigma = 2;

  constexpr static distribution_ranges<float> ranges{
      {1, 2}, {1, 2}, {1, 2}, {1, 2}};

  random_noise(points, sigma, ranges, UNIFORM, MAX_INTENSITY_255);

  for (tensor_size_t i = 0; i < points.size(0); i++) {
    EXPECT_GT(points[i].size(0), points_original[i].size(0))
        << "No noise has been added...";
  }
}

TEST(DeleteLabelsByMinPointsTest, BasicAssertions) {
  {
    const auto points = torch::tensor({{10.4966, 10.1144, 10.2182, -8.4158},
                                       {7.0241, 7.6908, -2.1535, 1.3416},
                                       {10.0, 10.0, 10.0, 10.0}});

    const torch::Tensor labels =
        torch::tensor({{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
                       {10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0}});
    const torch::Tensor names = torch::tensor({{0}, {1}});

    constexpr std::uint64_t min_points = 2;

    auto [result_labels, result_names] =
        _delete_labels_by_min_points(points, labels, names, min_points);

    const auto expected_points =
        torch::tensor({{10.4966, 10.1144, 10.2182, -8.4158},
                       {7.0241, 7.6908, -2.1535, 1.3416},
                       {10.0, 10.0, 10.0, 10.0}});

    const torch::Tensor expected_labels =
        torch::tensor({{10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0}});
    const torch::Tensor expected_names = torch::tensor({{1}});

    EXPECT_TRUE(points.equal(expected_points))
        << "Points should not have been modified!\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;
    EXPECT_TRUE(result_labels.equal(expected_labels))
        << "expected:\n"
        << expected_labels << "\nactual:\n"
        << labels;
    EXPECT_TRUE(result_names.equal(expected_names))
        << "expected:\n"
        << expected_names << "\nactual:\n"
        << names;
  }

  {

    const auto points = torch::tensor({{{-8.2224, -4.3151, -6.5488, -3.9899},
                                        {6.3092, -3.7737, 7.2516, -5.8651},
                                        {1.0, 1.0, 1.0, 10.0}},

                                       {{10.4966, 10.1144, 10.2182, -8.4158},
                                        {7.0241, 7.6908, -2.1535, 1.3416},
                                        {10.0, 10.0, 10.0, 10.0}}});

    const torch::Tensor labels =
        torch::tensor({{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
                        {100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 0.0}},
                       {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
                        {10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0}}});
    const torch::Tensor names =
        torch::tensor({{{0x00}, {0x01}}, {{0x10}, {0x11}}});

    const std::uint64_t min_points = 1;

    auto [result_labels, result_names] =
        delete_labels_by_min_points(points, labels, names, min_points);

    const auto expected_points =
        torch::tensor({{{-8.2224, -4.3151, -6.5488, -3.9899},
                        {6.3092, -3.7737, 7.2516, -5.8651},
                        {1.0, 1.0, 1.0, 10.0}},

                       {{10.4966, 10.1144, 10.2182, -8.4158},
                        {7.0241, 7.6908, -2.1535, 1.3416},
                        {10.0, 10.0, 10.0, 10.0}}});

    const auto expected_labels = torch::List<torch::Tensor>(
        {torch::tensor({{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0}}),
         torch::tensor({{10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0}})});
    const auto expected_names = torch::List<torch::Tensor>(
        {torch::tensor({{0x00}}), torch::tensor({{0x11}})});

    EXPECT_TRUE(points.equal(expected_points))
        << "Points should not have been modified!\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;

    ASSERT_EQ(result_labels.size(), expected_labels.size());

    for (std::size_t i = 0; i < result_labels.size(); i++) {
      const torch::Tensor result = result_labels[i];
      const torch::Tensor expected = expected_labels[i];

      EXPECT_TRUE(result.equal(expected)) << "expected:\n"
                                          << expected << "\nactual:\n"
                                          << result;
    }

    ASSERT_EQ(result_names.size(), expected_names.size());

    for (std::size_t i = 0; i < result_names.size(); i++) {
      const torch::Tensor result = result_names[i];
      const torch::Tensor expected = expected_names[i];

      EXPECT_TRUE(result.equal(expected)) << "expected:\n"
                                          << expected << "\nactual:\n"
                                          << result;
    }
  }
}

// doing tests with controlled random number generation (no random seed)
#ifdef TEST_RNG

TEST(DrawValuesTest, BasicAssertions) {
  std::uniform_int_distribution<int> ud(0, 100);
  std::normal_distribution<float> nd(0, 5);
  {
    auto ud_result = std::get<VECTOR>(draw_values<int>(ud, 10));
    std::vector<int> ud_expected_values{70, 72, 28, 43, 22, 69, 55, 72, 72, 49};

    auto nd_result = std::get<VECTOR>(draw_values<float>(nd, 10));
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
    auto ud_result = std::get<VALUE>(draw_values<int>(ud));
    int expected_value = 70;

    auto nd_result = std::get<VALUE>(draw_values<float>(nd));
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

    auto ud_result = std::get<VECTOR>(draw_values<int>(ud, {}, true));
    std::vector<int> ud_expected_value = {70};

    // NOTE(tom): for some reason this results in the second value from the
    // sequence, not the first
    auto nd_result = std::get<VECTOR>(draw_values<float>(nd, {}, true));
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

TEST(TranslateRandomTest, BasicAssertions) {

  auto points = torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                              torch::kF32);
  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}},
                              torch::kF32);
  const float sigma = 1;

  // NOTE(tom): the generated translation vector should be {1, 1, 1}
  translate_random(points, labels, sigma);

  auto expected_points = torch::tensor(
      {{{2.0, 3.0, 4.0, 10.0}, {5.0, 6.0, 7.0, -10.0}}}, torch::kF32);
  auto expected_labels = torch::tensor({{{2.0, 2.0, 2.0, 2.0, 3.0, 2.5, M_PI},
                                         {3.0, 3.0, 3.0, 1.0, 1.0, 0.5, M_PI}}},
                                       torch::kF32);

  EXPECT_TRUE(points.equal(expected_points));
  EXPECT_TRUE(labels.equal(expected_labels));
}

TEST(ScaleRandomTest, BasicAssertions) {

  auto points = torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                              torch::kF32);
  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}},
                              torch::kF32);
  const float sigma = 10;
  const float max_scale = 11;

  // NOTE(tom): scale_factor = 7.21812582
  scale_random(points, labels, sigma, max_scale);

  auto expected_points =
      torch::tensor({{{7.21812582, 14.43625164, 21.65437746, 10.0},
                      {28.87250328, 36.0906291, 43.30875492, -10.0}}},
                    torch::kF32);
  auto expected_labels =
      torch::tensor({{{7.21812582, 7.21812582, 7.21812582, 14.43625164,
                       21.65437746, 18.04531455, M_PI},
                      {14.43625164, 14.43625164, 14.43625164, 7.21812582,
                       7.21812582, 3.60906291, M_PI}}},
                    torch::kF32);

  EXPECT_TRUE(points.equal(expected_points));
  EXPECT_TRUE(labels.equal(expected_labels));
}

TEST(ScaleLocalTest, BasicAssertions) {

  {
    auto points = torch::tensor(
        {{{1.0, 2.0, 3.0, 10.0}, {100.0, 100.0, 100.0, -10.0}}}, torch::kF32);
    auto labels =
        torch::tensor({{{1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 0.0}}}, torch::kF32);

    constexpr float sigma = 10;
    constexpr float max_scale = 11;
    constexpr float scale_factor = 7.21812582;

    // NOTE(tom): scale_factor = 7.21812582
    scale_local(points, labels, sigma, max_scale);

    const auto expected_points = torch::tensor(
        {{{scale_factor * 1.0, scale_factor * 2.0, scale_factor * 3.0, 10.0},
          {100.0, 100.0, 100.0, -10.0}}},
        torch::kF32);
    const auto expected_labels =
        torch::tensor({{{1.0, 1.0, 1.0, scale_factor * 9.0, scale_factor * 9.0,
                         scale_factor * 9.0, 0.0}}},
                      torch::kF32);

    EXPECT_TRUE(points.equal(expected_points))
        << "points should be scaled: \nexpected" << expected_points
        << "\nactual:\n"
        << points;
    EXPECT_TRUE(labels.equal(expected_labels))
        << "label dimensions should be scaled: \nexpected" << expected_labels
        << "\nactual:\n"
        << labels;
  }
  {

    auto points = torch::tensor(
        {{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, torch::kF32);
    auto labels = torch::zeros({1, 0, 7});

    constexpr float sigma = 10;
    constexpr float max_scale = 11;

    // NOTE(tom): scale_factor = 7.21812582
    scale_local(points, labels, sigma, max_scale);

    const auto expected_points = torch::tensor(
        {{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, torch::kF32);
    const auto expected_labels = torch::zeros({1, 0, 7});

    EXPECT_TRUE(points.equal(expected_points))
        << "No scaling should happen since there are no labels!";
    EXPECT_TRUE(labels.equal(expected_labels))
        << "No scaling should happen since there are no labels!";
  }
}

TEST(FlipRandomTest, BasicAssertions) {
  {
    auto points = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
    auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.5},
                                  {2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5}}});

    static constexpr auto probability = 69;

    auto expected_points = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
    auto expected_labels =
        torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.5},
                        {2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5}}});

    flip_random(points, labels, probability);

    EXPECT_TRUE(points.equal(expected_points)) << "Points unexpectidly changed";
    EXPECT_TRUE(labels.equal(expected_labels)) << "Labels unexpectidly changed";
  }
  {
    auto points = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
    auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 2.5},
                                  {2.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5}}},
                                torch::kFloat16);

    static constexpr auto probability = 70;

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
}

TEST(RotateRandomTest, BasicAssertions) {

  auto points = torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                              torch::kF32);

  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}},
                              torch::kF32);

  constexpr float ROT_ANGLE = 28.0920143;
  rotate_random(points, labels, 50);

  auto points_coordinates =
      torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}}, torch::kF32);
  constexpr float angle = ROT_ANGLE * (M_PI / PI_DEG);
  const auto rotation_vector = rotate_yaw(angle);

  const auto v1 = torch::matmul(points_coordinates[0][0], rotation_vector);
  const auto v2 = torch::matmul(points_coordinates[0][1], rotation_vector);

  // The vectors need to contiguous for pointer pointer access
  ASSERT_TRUE(v1.is_contiguous());
  ASSERT_TRUE(v2.is_contiguous());

  const float *const vec1 = v1.const_data_ptr<float>();
  const float *const vec2 = v2.const_data_ptr<float>();

  const auto expected_points = torch::tensor(
      {{vec1[0], vec1[1], vec1[2], 10.0f}, {vec2[0], vec2[1], vec2[2], -10.0f}},
      torch::kF32);

  // NOTE(tom): Using allclose here because the equality check ends up failing
  //            despite there being no visible difference between the tensor
  //            elements. Even when printing with full 32bit floating point
  //            precision (%.9g).
  EXPECT_TRUE(points.allclose(expected_points))
      << "`points` not equal to `expected_points`:\npoints:" << points
      << "\nexpected_points:\n"
      << expected_points;

  auto label_coordinates =
      torch::tensor({{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}}}, torch::kF32);

  const auto l1 = torch::matmul(label_coordinates[0][0], rotation_vector);
  const auto l2 = torch::matmul(label_coordinates[0][1], rotation_vector);

  // The vectors need to contiguous for pointer pointer access
  ASSERT_TRUE(l1.is_contiguous());
  ASSERT_TRUE(l2.is_contiguous());

  const float *const label1 = l1.const_data_ptr<float>();
  const float *const label2 = l2.const_data_ptr<float>();

  const float angle_label1 = fmodf32((M_PI + angle), (2.0f * M_PI));
  const float angle_label2 = fmodf32((M_PI + angle), (2.0f * M_PI));

  const auto expected_labels = torch::tensor(
      {{label1[0], label1[1], label1[2], 2.0f, 3.0f, 2.5f, angle_label1},
       {label2[0], label2[1], label2[2], 1.0f, 1.0f, 0.5f, angle_label2}},
      torch::kF32);

  // NOTE(tom): Using allclose here because the equality check ends up failing
  //            despite there being no visible difference between the tensor
  //            elements. Even when printing with full 32bit floating point
  //            precision (%.9g).
  EXPECT_TRUE(labels.allclose(expected_labels))
      << "`labels` not equal to `expected_labels`:\nlabels:" << labels
      << "\nexpected_labels:\n"
      << expected_labels;
}

TEST(ThinOutTest, BasicAssertions) {
  constexpr tensor_size_t BATCHES = 2;
  constexpr tensor_size_t ITEMS = 10;

  const auto points = torch::rand({BATCHES, ITEMS, 4}, torch::kF32);
  auto new_points = thin_out(points, 1);

  // NOTE(tom): percent = 0.653750002, indices = {9, 4, 8, 0}
  const auto indices = torch::tensor({9, 4, 8, 0});

  const auto expected_points = points.index_select(1, indices);

  EXPECT_EQ(BATCHES, points.size(0)) << "`points` was not supposed to change!";
  EXPECT_EQ(ITEMS, points.size(1)) << "`points` was not supposed to change!";

  EXPECT_EQ(BATCHES, expected_points.size(0))
      << "Batch dimensions were not supposed to change!";
  EXPECT_EQ(BATCHES, new_points.size(0))
      << "Batch dimensions were not supposed to change!";

  EXPECT_EQ(new_points.size(1), expected_points.size(1))
      << "Number of points does not match!";

  EXPECT_TRUE(new_points.equal(expected_points))
      << "Thin out not as expected:\noriginal:\n"
      << points << "\nexpected:\n"
      << expected_points << "\nactual:\n"
      << new_points;
}

TEST(RandomPointNoiseTest, BasicAssertions) {
  constexpr float sigma = 1;

  auto points = torch::tensor({{{1.0, 2.0, 3.0, 4.0}, {-1.0, -2.0, -3.0, -4.0}},
                               {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}});

  // noise_vector = {{1.08580697, 1.00174832, -0.566087246},
  //                {-1.69251204, 0.729757607, -1.04425383}};
  // NOTE(tom): The RNG is not reset after each iteration of the inner loop.
  //            It only resets after each iteration of the outer loop.
  const auto expected_points =
      torch::tensor({{{2.08580697, 3.00174832, 2.433912754, 4.0},
                      {-2.69251204, -1.270242393, -4.04425383, -4.0}},
                     {{2.08580697, 2.00174832, 0.433912754, 0.0},
                      {-1.69251204, 0.729757607, -0.04425383, 1.0}}});

  random_point_noise(points, sigma);

  // NOTE(tom): This currently fails because `draw_values` is behaving weirdly.
  EXPECT_TRUE(points.allclose(expected_points))
      << "Noise not as expected:\nexpected:\n"
      << expected_points << "\nactual:\n"
      << points;
}

TEST(TransformAlongRayTest, BasicAssertions) {
  constexpr float sigma = 1;

  auto points = torch::tensor({{{1.0, 2.0, 3.0, 4.0}, {-1.0, -2.0, -3.0, -4.0}},
                               {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}});

  // noise_value = 1.08580697, 1.00174832;
  // NOTE(tom): The RNG is not reset after each iteration of the inner loop.
  //            It only resets after each iteration of the outer loop.
  const auto expected_points =
      torch::tensor({{{2.08580697, 3.08580697, 4.08580697, 4.0},
                      {0.00174832, -0.99825168, -1.99825168, -4.0}},
                     {{2.08580697, 2.08580697, 2.08580697, 0.0},
                      {1.00174832, 1.00174832, 2.00174832, 1.0}}});

  transform_along_ray(points, sigma);
  EXPECT_TRUE(points.allclose(expected_points))
      << "Transformation not as expected:\nexpected:\n"
      << expected_points << "\nactual:\n"
      << points;
}

TEST(IntensityNoiseTest, BasicAssertions) {
  {
    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                       {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});

    constexpr float SIGMA = 20;
    constexpr intensity_range MAX_INTENSITY = MAX_INTENSITY_255;

    // NOTE(tom): value of intensity_shift = 21.2925453
    const auto expected_points =
        torch::tensor({{{1.0, 2.0, 3.0, 25.7925453}, {-1.0, -2.0, -3.0, 255.0}},
                       {{1.0, 1.0, 1.0, 21.2925453}, {0.0, 0.0, 1.0, 255.0}}});

    intensity_noise(points, SIGMA, MAX_INTENSITY);

    EXPECT_TRUE(points.allclose(expected_points))
        << "Noise values not as expected:\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;
  }
  {
    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 0.52}, {-1.0, -2.0, -3.0, 1.0}},
                       {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.95}}});

    constexpr float SIGMA = 0.2;
    constexpr intensity_range MAX_INTENSITY = MAX_INTENSITY_1;

    // NOTE(tom): value of intensity_shift = 0.212925255
    const auto expected_points =
        torch::tensor({{{1.0, 2.0, 3.0, 0.732925255}, {-1.0, -2.0, -3.0, 1.0}},
                       {{1.0, 1.0, 1.0, 0.212925255}, {0.0, 0.0, 1.0, 1.0}}});

    intensity_noise(points, SIGMA, MAX_INTENSITY);

    EXPECT_TRUE(points.allclose(expected_points))
        << "Noise values not as expected:\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;
  }
}

TEST(IntensityShiftTest, BasicAssertions) {
  {
    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                       {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});

    constexpr float SIGMA = 20;
    constexpr intensity_range MAX_INTENSITY = MAX_INTENSITY_255;

    // NOTE(tom): value of intensity_shift = 21.2925453
    const auto expected_points =
        torch::tensor({{{1.0, 2.0, 3.0, 25.7925453}, {-1.0, -2.0, -3.0, 255.0}},
                       {{1.0, 1.0, 1.0, 21.2925453}, {0.0, 0.0, 1.0, 255.0}}});

    intensity_shift(points, SIGMA, MAX_INTENSITY);

    EXPECT_TRUE(points.allclose(expected_points))
        << "Noise shift not as expected:\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;
  }
  {
    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 0.52}, {-1.0, -2.0, -3.0, 1.0}},
                       {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 0.95}}});

    constexpr float SIGMA = 0.2;
    constexpr intensity_range MAX_INTENSITY = MAX_INTENSITY_1;

    // NOTE(tom): value of intensity_shift = 0.212925255
    const auto expected_points =
        torch::tensor({{{1.0, 2.0, 3.0, 0.732925255}, {-1.0, -2.0, -3.0, 1.0}},
                       {{1.0, 1.0, 1.0, 0.212925255}, {0.0, 0.0, 1.0, 1.0}}});

    intensity_shift(points, SIGMA, MAX_INTENSITY);

    EXPECT_TRUE(points.allclose(expected_points))
        << "Noise shift not as expected:\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;
  }
}

#endif

// NOLINTEND

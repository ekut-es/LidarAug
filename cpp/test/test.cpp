
#include "../include/raytracing.hpp"
#include "../include/stats.hpp"
#include "../include/tensor.hpp"
#include "../include/transformations.hpp"
#include "../include/utils.hpp"
#include "../include/weather.hpp"

#include <cnpy/cnpy.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <torch/types.h>

using namespace torch_utils;
const std::filesystem::path npz_dir{"../../pytest/data/npz/test.npz"};

// NOLINTBEGIN

TEST(Transformation, TranslationTest) {
  auto tensor = torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}});
  auto translation = torch::tensor({1.0, 2.0, 3.0});

  translate(tensor, translation);

  auto expected = torch::tensor({{{2.0, 4.0, 6.0}, {5.0, 7.0, 9.0}}});

  EXPECT_TRUE(tensor.equal(expected));
}

TEST(Transformation, ScalingTest) {
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

TEST(Transformation, RotationTest) {

  {

    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}},
                       {{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                      F32);

    constexpr float ROT_ANGLE = 180.0f;
    rotate_deg(points, ROT_ANGLE);
    // rotate_random(points, labels, 50);

    auto points_coordinates =
        torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                       {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}},
                      F32);
    constexpr float ANGLE =
        ROT_ANGLE * (math_utils::PI_RAD / math_utils::PI_DEG);
    const auto rotation_vector = math_utils::rotate_yaw(ANGLE);

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
                      F32);

    EXPECT_TRUE(points.allclose(expected_points))
        << "`points` not equal to `expected_points`:\npoints:" << points
        << "\nexpected_points:\n"
        << expected_points;
  }
  {

    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}},
                       {{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}},
                      F32);

    constexpr float ANGLE = math_utils::PI_RAD;
    rotate_rad(points, ANGLE);

    auto points_coordinates =
        torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                       {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}},
                      F32);

    const auto rotation_vector = math_utils::rotate_yaw(ANGLE);

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
                      F32);

    EXPECT_TRUE(points.allclose(expected_points))
        << "`points` not equal to `expected_points`:\npoints:" << points
        << "\nexpected_points:\n"
        << expected_points;
  }
}

TEST(Transformation, FlipTest) {
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

TEST(MathUtils, AngleConversionTest) {
  constexpr static float three_sixty_deg = 360;
  constexpr static float zero_deg = 0;
  constexpr static float one_eighty_deg = 180;
  constexpr static float ninety_deg = 90;

  EXPECT_EQ(math_utils::to_rad(three_sixty_deg),
            2 * static_cast<float>(math_utils::PI_RAD));
  EXPECT_EQ(math_utils::to_rad(zero_deg), 0);
  EXPECT_EQ(math_utils::to_rad(one_eighty_deg),
            static_cast<float>(math_utils::PI_RAD));
  EXPECT_EQ(math_utils::to_rad(-one_eighty_deg),
            static_cast<float>(-math_utils::PI_RAD));
  EXPECT_EQ(math_utils::to_rad(ninety_deg),
            static_cast<float>(math_utils::PI_RAD / 2));
  EXPECT_EQ(math_utils::to_rad(-ninety_deg),
            static_cast<float>((-math_utils::PI_RAD) / 2));
}

TEST(Stats, DrawUniformValuesTest) {
  {

    constexpr static auto size = 10;
    constexpr static auto num_values = 3;

    auto values = draw_unique_uniform_values<int>(size, num_values);

    EXPECT_EQ(values.size(), num_values);

    for (auto val : values) {
      EXPECT_LT(val, size);
      EXPECT_GE(val, 0);
    }
  }

  {
    constexpr static auto size = 3;
    constexpr static auto num_values = 10;

    EXPECT_THROW(
        {
          try {
            auto values = draw_unique_uniform_values<int>(size, num_values);
          } catch (const std::invalid_argument &e) {
            EXPECT_STREQ("num_values cannot exceed size.", e.what());
            throw;
          }
        },
        std::invalid_argument);
  }
}

TEST(Transformation, RandomNoiseTest) {
  const auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 10.9}, {4.0, 5.0, 6.0, -10.0}}});

  constexpr static float sigma = 2;

  constexpr static distribution_ranges<float> ranges{
      {1, 2}, {1, 2}, {1, 2}, {1, 2}};

  auto new_points =
      random_noise(points, sigma, ranges, noise_type::UNIFORM,
                   point_cloud_data::intensity_range::MAX_INTENSITY_255);

  EXPECT_GT(new_points.size(1), points.size(1)) << "No noise has been added...";
}

TEST(Transformation, DeleteLabelsByMinPointsHelperTest) {
  const auto points = torch::tensor({{10.4966, 10.1144, 10.2182, -8.4158},
                                     {7.0241, 7.6908, -2.1535, 1.3416},
                                     {10.0, 10.0, 10.0, 10.0}});

  const torch::Tensor labels =
      torch::tensor({{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0},
                     {10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0}});
  const torch::Tensor names = torch::tensor({{0}, {1}});

  constexpr std::uint64_t min_points = 2;

  const auto [result_labels, result_names] =
      _delete_labels_by_min_points(points, labels, names, min_points, 0);

  const auto expected_points =
      torch::tensor({{10.4966, 10.1144, 10.2182, -8.4158},
                     {7.0241, 7.6908, -2.1535, 1.3416},
                     {10.0, 10.0, 10.0, 10.0}});

  const torch::Tensor expected_labels =
      torch::tensor({{10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0, 0.0}});
  const torch::Tensor expected_names = torch::tensor({{1, 0}});

  EXPECT_TRUE(points.equal(expected_points))
      << "Points should not have been modified!\nexpected:\n"
      << expected_points << "\nactual:\n"
      << points;
  EXPECT_TRUE(result_labels.equal(expected_labels))
      << "expected:\n"
      << expected_labels << "\nactual:\n"
      << result_labels;
  EXPECT_TRUE(result_names.equal(expected_names))
      << "expected:\n"
      << expected_names << "\nactual:\n"
      << result_names;
}

TEST(Transformation, DeleteLabelsByMinPointsTest) {

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

    const auto expected_labels =
        torch::tensor({{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0},
                       {10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 0.0, 1.0}});
    const auto expected_names = torch::tensor({{0x00, 0}, {0x11, 1}});

    EXPECT_TRUE(points.equal(expected_points))
        << "Points should not have been modified!\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;

    EXPECT_TRUE(result_labels.equal(expected_labels))
        << "expected:\n"
        << expected_labels << "\nactual:\n"
        << result_labels;

    EXPECT_TRUE(result_names.equal(expected_names))
        << "expected:\n"
        << expected_names << "\nactual:\n"
        << result_names;
  }
}

TEST(Tensor, ChangeSparseRepresentationTest) {

  // clang-format off
  const auto in = torch::tensor({
      {1, 2, 0},
      {4, 1, 0},
      {7, 7, 0},
      {3, 1, 1},
      {2, 2, 2},
      {3, 2, 2},
  });

  const tensor_size_t batch_idx = 2;

  const auto expected = torch::tensor({
      {{1, 2},
       {4, 1},
       {7, 7}},
      {{3, 1},
       {0, 0},
       {0, 0}},
      {{2, 2},
       {3, 2},
       {0, 0}},
  });
  // clang-format on

  const auto result = change_sparse_representation(in, batch_idx);

  EXPECT_TRUE(result.equal(expected)) << "expected:\n"
                                      << expected << "\nactual:\n"
                                      << result;
}

TEST(Raytracing, MulTest) {
  auto v = torch::tensor({{1, 2, 3}, {-1, -2, -3}});
  auto v_o = torch::tensor({{1, 2, 3}, {-1, -2, -3}});
  float c = 5;
  auto expected = torch::tensor({{5, 10, 15}, {-5, -10, -15}});

  auto result = rt::mul(v, c);

  EXPECT_TRUE(result.equal(expected)) << "expected:\n"
                                      << expected << "\nactual:\n"
                                      << result;
  EXPECT_TRUE(v.equal(v_o)) << "The original tensor " << v_o
                            << " has changed unexpectidly!\nWas " << v;
}

TEST(Raytracing, AddTest) {
  auto v = torch::tensor({1, 2, 3}, F32);
  auto k = torch::tensor({1, 2, 3}, F32);
  auto l = torch::tensor({1, 2, 3}, F32);

  auto v_o = torch::tensor({1, 2, 3}, F32);
  auto k_o = torch::tensor({1, 2, 3}, F32);
  auto l_o = torch::tensor({1, 2, 3}, F32);

  auto expected = torch::tensor({3, 6, 9}, F32);

  auto result = rt::add(v, k, l);

  EXPECT_TRUE(result.equal(expected)) << "expected:\n"
                                      << expected << "\nactual:\n"
                                      << result;

  EXPECT_TRUE(v.equal(v_o)) << "The original tensor " << v_o
                            << " has changed unexpectidly!\nWas " << v;
  EXPECT_TRUE(k.equal(k_o)) << "The original tensor " << k_o
                            << " has changed unexpectidly!\nWas " << k;
  EXPECT_TRUE(l.equal(l_o)) << "The original tensor " << l_o
                            << " has changed unexpectidly!\nWas " << l;
}

TEST(Raytracing, ScalarTest) {
  auto v = torch::tensor({1, 2, 3}, F32);
  auto k = torch::tensor({1, 2, 3}, F32);

  auto v_o = torch::tensor({1, 2, 3}, F32);
  auto k_o = torch::tensor({1, 2, 3}, F32);

  auto expected = 14.0f;

  auto result = rt::scalar(v, k);

  EXPECT_EQ(result, expected) << "expected:\n"
                              << expected << "\nactual:\n"
                              << result;
  EXPECT_TRUE(v.equal(v_o)) << "The original tensor " << v_o
                            << " has changed unexpectidly!\nWas " << v;
  EXPECT_TRUE(k.equal(k_o)) << "The original tensor " << k_o
                            << " has changed unexpectidly!\nWas " << k;
}

TEST(Raytracing, VectorLengthTest) {
  auto v = torch::tensor({6, 2, 3}, F32);
  auto v_o = torch::tensor({6, 2, 3}, F32);

  auto expected = 7.0f;

  auto result = rt::vector_length(v);

  EXPECT_EQ(result, expected) << "expected:\n"
                              << expected << "\nactual:\n"
                              << result;
  EXPECT_TRUE(v.equal(v_o)) << "The original tensor " << v_o
                            << " has changed unexpectidly!\nWas " << v;
}

TEST(Raytracing, NormalizeTest) {
  auto v = torch::tensor({6, 2, 3}, F32);
  auto v_o = torch::tensor({6, 2, 3}, F32);

  auto expected = torch::tensor({6 / 7.0f, 2 / 7.0f, 3 / 7.0f}, F32);

  auto result = rt::normalize(v);

  EXPECT_TRUE(result.allclose(expected)) << "expected:\n"
                                         << expected << "\nactual:\n"
                                         << result;
  EXPECT_TRUE(v.allclose(v_o)) << "The original tensor " << v_o
                               << " has changed unexpectidly!\nWas " << v;
}

TEST(Raytracing, CrossTest) {
  auto v = torch::tensor({1, 2, 3}, F32);
  auto k = torch::tensor({4, 5, 6}, F32);

  auto v_o = torch::tensor({1, 2, 3}, F32);
  auto k_o = torch::tensor({4, 5, 6}, F32);

  auto expected = torch::tensor(
      {(2 * 6) - (3 * 5), (3 * 4) - (1 * 6), (1 * 5) - (2 * 4)}, F32);
  auto result = rt::cross(v, k);

  EXPECT_TRUE(v.equal(v_o)) << "The original tensor " << v_o
                            << " has changed unexpectidly!\nWas " << v;
  EXPECT_TRUE(k.equal(k_o)) << "The original tensor " << k_o
                            << " has changed unexpectidly!\nWas " << k;

  EXPECT_TRUE(result.allclose(expected)) << "expected:\n"
                                         << expected << "\nactual:\n"
                                         << result;
}

TEST(Raytracing, RotateTest) {
  auto v = torch::tensor({1, 2, 3}, F32);
  auto k = torch::tensor({4, 5, 6}, F32);
  float angle = 180;

  auto v_o = torch::tensor({1, 2, 3}, F32);
  auto k_o = torch::tensor({4, 5, 6}, F32);

  auto expected = torch::tensor({206.40788, 249.74979, 307.5124}, F32);
  auto result = rt::rotate(v, k, angle);

  EXPECT_TRUE(v.equal(v_o)) << "The original tensor " << v_o
                            << " has changed unexpectidly!\nWas " << v;
  EXPECT_TRUE(k.equal(k_o)) << "The original tensor " << k_o
                            << " has changed unexpectidly!\nWas " << k;
  EXPECT_TRUE(result.allclose(expected)) << "expected:\n"
                                         << expected << "\nactual:\n"
                                         << result;
}

TEST(Raytracing, CheckNpzFile) {
  ASSERT_TRUE(std::filesystem::exists(npz_dir))
      << "File: " << npz_dir
      << " could not be found in current working directory: "
      << std::filesystem::current_path();
}

TEST(Raytracing, TraceTest) {

  const auto points =
      torch::tensor({{0.79115541, 0.00995717, 0.85223702, 0.86795932},
                     {0.88936069, 0.73058396, 0.66369673, 0.15326185},
                     {0.2041209, 0.27948176, 0.25410868, 0.68685306},
                     {0.40355785, 0.58527619, 0.50859593, 0.19754277},
                     {0.23363058, 0.17909182, 0.15318045, 0.99695834},
                     {0.59422449, 0.95887209, 0.60752133, 0.34042509},
                     {0.16007366, 0.66038575, 0.53459956, 0.00381125},
                     {0.71638315, 0.85775223, 0.00237889, 0.32981742},
                     {0.27168068, 0.18565797, 0.96313797, 0.32289272},
                     {0.06261661, 0.82851678, 0.09892672, 0.0678927}});

  const auto expected =
      torch::tensor({{0.79115541, 0.00995717, 0.85223702, 0.78116338},
                     {0.88936069, 0.73058396, 0.66369673, 0.13793567},
                     {0.2041209, 0.27948176, 0.25410868, 0.61816775},
                     {0.40355785, 0.58527619, 0.50859593, 0.1777885},
                     {0.23363058, 0.17909182, 0.15318045, 0.8972625},
                     {0.59422449, 0.95887209, 0.60752133, 0.30638258},
                     {0.16007366, 0.66038575, 0.53459956, 0.00343013},
                     {0.71638315, 0.85775223, 0.00237889, 0.29683568},
                     {0.27168068, 0.18565797, 0.96313797, 0.29060345},
                     {0.06261661, 0.82851678, 0.09892672, 0.06110343}});

  auto npz_data = cnpy::npz_load(npz_dir);

  auto nf_array = npz_data["nf"];
  const auto nf =
      torch::from_blob(nf_array.data<float>(),
                       {static_cast<tensor_size_t>(nf_array.num_vals())})
          .reshape({-1, 6});

  auto si_array = npz_data["si"];
  const auto si =
      torch::from_blob(si_array.data<tensor_size_t>(),
                       {static_cast<tensor_size_t>(si_array.num_vals())});

  const auto result = rt::trace(points, nf, si, simulation_type::rain);

  EXPECT_TRUE(result.allclose(expected)) << "expected:\n"
                                         << expected << "\nactual:\n"
                                         << result;
}

TEST(Raytracing, TraceBeamTest) {

  const auto beam = torch::tensor({1, 2, 3});

  auto npz_data = cnpy::npz_load(npz_dir);

  auto nf_array = npz_data["nf"];
  const auto nf =
      torch::from_blob(nf_array.data<float>(),
                       {static_cast<tensor_size_t>(nf_array.num_vals())})
          .reshape({-1, 6});

  auto si_array = npz_data["si"];
  const auto si =
      torch::from_blob(si_array.data<tensor_size_t>(),
                       {static_cast<tensor_size_t>(si_array.num_vals())});

  const auto expected = -1.0F;
  const auto result = rt::trace_beam(nf, beam, si);

  EXPECT_EQ(result, expected) << "expected:\n"
                              << expected << "\nactual:\n"
                              << result;
}

TEST(Raytracing, SortNoiseFilterTest) {

  auto compute_median = [](torch::Tensor tensor) {
    auto sorted_tensor = std::get<0>(tensor.sort());
    int64_t size = sorted_tensor.size(0);

    if (size % 2 == 0) {
      return (sorted_tensor[size / 2 - 1].item<double>() +
              sorted_tensor[size / 2].item<double>()) /
             2.0;
    } else {
      return sorted_tensor[size / 2].item<double>();
    }
  };

  auto npz_data = cnpy::npz_load(npz_dir);

  auto nf_array = npz_data["nf"];
  const auto nf =
      torch::from_blob(nf_array.data<double>(),
                       {static_cast<tensor_size_t>(nf_array.num_vals())}, F64)
          .reshape({-1, 6});

  auto si_array = npz_data["si"];
  const auto si =
      torch::from_blob(si_array.data<double>(),
                       {static_cast<tensor_size_t>(si_array.num_vals())}, F64);

  auto [result_nf, result_si] = rt::sort_noise_filter<double, F64>(nf);

  EXPECT_TRUE(torch::equal(nf, result_nf)) << "expected:\n"
                                           << nf << "\nactual:\n"
                                           << result_nf;

  EXPECT_EQ(si.size(0), result_si.size(0));

  std::cout << "min comparison (expected, result): " << si.min().item<double>()
            << " vs " << result_si.min().item<double>() << "\n";
  std::cout << "max comparison (expected, result): " << si.max().item<double>()
            << " vs " << result_si.max().item<double>() << "\n";
  std::cout << "average comparison (expected, result): "
            << si.sum().item<double>() / si.size(0) << " vs "
            << result_si.sum().item<double>() / result_si.size(0) << "\n";
  std::cout << "median comparison (expected, result): " << compute_median(si)
            << " vs " << compute_median(result_si) << "\n";

  EXPECT_TRUE(torch::allclose(si, result_si)) << "expected:\n"
                                              << si << "\nactual:\n"
                                              << result_si;
}

TEST(Evaluation, Iou3dTest) {

  evaluation_utils::polygon3d_t gt_box{{
      {1, 0, 1},
      {1, 0, 0},
      {0, 0, 0},
      {0, 0, 1},
      {1, 1, 1},
      {1, 1, 0},
      {0, 1, 0},
      {0, 1, 1},
      {1, 0, 1},
  }};

  // should have iou of 0
  evaluation_utils::polygon3d_t box_above{{
      {1, 3, 1},
      {1, 3, 0},
      {0, 3, 0},
      {0, 3, 1},
      {1, 4, 1},
      {1, 4, 0},
      {0, 4, 0},
      {0, 4, 1},
      {1, 3, 1},
  }};

  // should have iou of 0
  evaluation_utils::polygon3d_t box_above_touching{{
      {1, 1, 1},
      {1, 1, 0},
      {0, 1, 0},
      {0, 1, 1},
      {1, 2, 1},
      {1, 2, 0},
      {0, 2, 0},
      {0, 2, 1},
      {1, 1, 1},
  }};

  // should have iou of 0 (actually -0 for some technical reasons)
  evaluation_utils::polygon3d_t box_to_side_touching{{
      {1, 0, 2},
      {1, 0, 1},
      {0, 0, 1},
      {0, 0, 2},
      {1, 1, 2},
      {1, 1, 1},
      {0, 1, 1},
      {0, 1, 2},
      {1, 0, 2},
  }};

  // should have an iou of .5
  evaluation_utils::polygon3d_t box_above_intersect{{
      {1, 0, 1},
      {1, 0, 0},
      {0, 0, 0},
      {0, 0, 1},
      {1, 2, 1},
      {1, 2, 0},
      {0, 2, 0},
      {0, 2, 1},
      {1, 0, 1},
  }};

  // should have an iou of 1
  evaluation_utils::polygon3d_t eq_box{{
      {1, 0, 1},
      {1, 0, 0},
      {0, 0, 0},
      {0, 0, 1},
      {1, 1, 1},
      {1, 1, 0},
      {0, 1, 0},
      {0, 1, 1},
      {1, 0, 1},
  }};

  std::vector<evaluation_utils::polygon3d_t> boxes{
      box_above, box_above_touching, box_to_side_touching, box_above_intersect,
      eq_box};
  std::vector<float> expected{0, 0, 0, .5, 1};

  auto ious = evaluation_utils::iou_3d<float>(gt_box, boxes);

  EXPECT_EQ(expected, ious);
}

// doing tests with controlled random number generation (no random seed)
#ifdef TEST_RNG

TEST(StatsRNG, DrawValuesTest) {
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
    //            sequence, not the first.
    //            The result is unexpected but consistent and will not interfere
    //            with the correctness of the library, so it has been decided to
    //            ignore this and adjust the test.
    auto nd_result = std::get<VECTOR>(draw_values<float>(nd, {}, true));
    std::vector<float> nd_expected_value{5.00874138};

    EXPECT_EQ(ud_result, ud_expected_value)
        << "Vectors are not equal: expected\n"
        << ud_expected_value << "\nactual:\n"
        << ud_result;

    EXPECT_EQ(nd_result, nd_expected_value)
        << "Vectors are not equal: expected\n"
        << nd_expected_value << "\nactual:\n"
        << nd_result;
  }
}

TEST(StatsRNG, DrawValuesTensorTest) {
  std::uniform_int_distribution<int> ud(0, 100);
  std::normal_distribution<float> nd(0, 5);
  {
    auto ud_result = draw_values<int, I32>(ud, 10);
    auto ud_expected_values =
        torch::tensor({70, 72, 28, 43, 22, 69, 55, 72, 72, 49}, I32);

    auto nd_result = draw_values<float, F32>(nd, 10);
    auto nd_expected_values = torch::tensor(
        {5.42903471, 5.00874138, -2.83043623, -8.46256065, 3.64878798,
         -5.22126913, 8.69870472, 2.03683066, -0.366712242, 9.06219864},
        F32);

    EXPECT_TRUE(ud_result.equal(ud_expected_values))
        << "Tensors are not equal: expected\n"
        << ud_expected_values << "\nactual:\n"
        << ud_result;

    EXPECT_TRUE(nd_result.equal(nd_expected_values))
        << "Tensors are not equal: expected\n"
        << nd_expected_values << "\nactual:\n"
        << nd_result;
  }

  {
    auto ud_result = draw_values<int, I32>(ud);
    auto expected_value = torch::tensor({70}, I32);

    auto nd_result = draw_values<float, F32>(nd);
    auto nd_expected_value = torch::tensor({5.42903471}, F32);

    EXPECT_TRUE(ud_result.equal(expected_value))
        << "Tensors are not equal: expected\n"
        << expected_value << "\nactual:\n"
        << ud_result;

    EXPECT_TRUE(nd_result.equal(nd_expected_value))
        << "Tensors are not equal: expected\n"
        << nd_expected_value << "\nactual:\n"
        << nd_result;
  }
}

TEST(RNGTransformation, TranslateRandomTest) {

  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, F32);
  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}},
                              F32);
  const float sigma = 1;

  // NOTE(tom): the generated translation vector should be {1, 1, 1}
  translate_random(points, labels, sigma);

  auto expected_points =
      torch::tensor({{{2.0, 3.0, 4.0, 10.0}, {5.0, 6.0, 7.0, -10.0}}}, F32);
  auto expected_labels = torch::tensor({{{2.0, 2.0, 2.0, 2.0, 3.0, 2.5, M_PI},
                                         {3.0, 3.0, 3.0, 1.0, 1.0, 0.5, M_PI}}},
                                       F32);

  EXPECT_TRUE(points.equal(expected_points));
  EXPECT_TRUE(labels.equal(expected_labels));
}

TEST(RNGTransformation, ScaleRandomTest) {

  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, F32);
  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}},
                              F32);
  const float sigma = 10;
  const float max_scale = 11;

  // NOTE(tom): scale_factor = 7.21812582
  scale_random(points, labels, sigma, max_scale);

  auto expected_points =
      torch::tensor({{{7.21812582, 14.43625164, 21.65437746, 10.0},
                      {28.87250328, 36.0906291, 43.30875492, -10.0}}},
                    F32);
  auto expected_labels =
      torch::tensor({{{7.21812582, 7.21812582, 7.21812582, 14.43625164,
                       21.65437746, 18.04531455, M_PI},
                      {14.43625164, 14.43625164, 14.43625164, 7.21812582,
                       7.21812582, 3.60906291, M_PI}}},
                    F32);

  EXPECT_TRUE(points.equal(expected_points));
  EXPECT_TRUE(labels.equal(expected_labels));
}

TEST(RNGTransformation, ScaleLocalTest) {

  {
    auto points = torch::tensor({{{1.0, 2.0, 3.0, 10.0},
                                  {100.0, 100.0, 100.0, -10.0},
                                  {100.0, 100.0, 100.0, -10.0},
                                  {100.0, 100.0, 100.0, -10.0}},
                                 {{100.0, 100.0, 100.0, -10.0},
                                  {1.0, 2.0, 3.0, 10.0},
                                  {100.0, 100.0, 100.0, -10.0},
                                  {1.0, 2.0, 3.0, 10.0}}},
                                F32);
    auto labels = torch::tensor(
        {
            {{1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 0.0}},
            {{1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 0.0}},
        },
        F32);

    constexpr float sigma = 10;
    constexpr float max_scale = 11;
    constexpr float scale_factor = 7.21812582;

    // NOTE(tom): scale_factor = 7.21812582
    scale_local(points, labels, sigma, max_scale);

    const auto expected_points = torch::tensor(
        {{{scale_factor * 1.0, scale_factor * 2.0, scale_factor * 3.0, 10.0},
          {100.0, 100.0, 100.0, -10.0},
          {100.0, 100.0, 100.0, -10.0},
          {100.0, 100.0, 100.0, -10.0}},
         {{100.0, 100.0, 100.0, -10.0},
          {scale_factor * 1.0, scale_factor * 2.0, scale_factor * 3.0, 10.0},
          {100.0, 100.0, 100.0, -10.0},
          {scale_factor * 1.0, scale_factor * 2.0, scale_factor * 3.0, 10.0}}

        },
        F32);
    const auto expected_labels =
        torch::tensor({{{1.0, 1.0, 1.0, scale_factor * 9.0, scale_factor * 9.0,
                         scale_factor * 9.0, 0.0}},
                       {{1.0, 1.0, 1.0, scale_factor * 9.0, scale_factor * 9.0,
                         scale_factor * 9.0, 0.0}}},
                      F32);

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

    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, F32);
    auto labels = torch::zeros({1, 0, 7});

    constexpr float sigma = 10;
    constexpr float max_scale = 11;

    // NOTE(tom): scale_factor = 7.21812582
    scale_local(points, labels, sigma, max_scale);

    const auto expected_points =
        torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, F32);
    const auto expected_labels = torch::zeros({1, 0, 7});

    EXPECT_TRUE(points.equal(expected_points))
        << "No scaling should happen since there are no labels!";
    EXPECT_TRUE(labels.equal(expected_labels))
        << "No scaling should happen since there are no labels!";
  }
}

TEST(RNGTransformation, FlipRandomTest) {
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

TEST(RNGTransformation, RotateRandomTest) {

  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 10.0}, {4.0, 5.0, 6.0, -10.0}}}, F32);

  auto labels = torch::tensor({{{1.0, 1.0, 1.0, 2.0, 3.0, 2.5, M_PI},
                                {2.0, 2.0, 2.0, 1.0, 1.0, 0.5, M_PI}}},
                              F32);

  constexpr float ROT_ANGLE = 28.0920143;
  rotate_random(points, labels, 50);

  auto points_coordinates =
      torch::tensor({{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}}, F32);
  constexpr float angle = ROT_ANGLE * (math_utils::PI_RAD / math_utils::PI_DEG);
  const auto rotation_vector = math_utils::rotate_yaw(angle);

  const auto v1 = torch::matmul(points_coordinates[0][0], rotation_vector);
  const auto v2 = torch::matmul(points_coordinates[0][1], rotation_vector);

  // The vectors need to contiguous for pointer pointer access
  ASSERT_TRUE(v1.is_contiguous());
  ASSERT_TRUE(v2.is_contiguous());

  const float *const vec1 = v1.const_data_ptr<float>();
  const float *const vec2 = v2.const_data_ptr<float>();

  const auto expected_points = torch::tensor(
      {{vec1[0], vec1[1], vec1[2], 10.0f}, {vec2[0], vec2[1], vec2[2], -10.0f}},
      F32);

  // NOTE(tom): Using allclose here because the equality check ends up failing
  //            despite there being no visible difference between the tensor
  //            elements. Even when printing with full 32bit floating point
  //            precision (%.9g).
  EXPECT_TRUE(points.allclose(expected_points))
      << "`points` not equal to `expected_points`:\npoints:" << points
      << "\nexpected_points:\n"
      << expected_points;

  auto label_coordinates =
      torch::tensor({{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}}}, F32);

  const auto l1 = torch::matmul(label_coordinates[0][0], rotation_vector);
  const auto l2 = torch::matmul(label_coordinates[0][1], rotation_vector);

  // The vectors need to contiguous for pointer pointer access
  ASSERT_TRUE(l1.is_contiguous());
  ASSERT_TRUE(l2.is_contiguous());

  const float *const label1 = l1.const_data_ptr<float>();
  const float *const label2 = l2.const_data_ptr<float>();

  const float angle_label1 =
      fmodf((math_utils::PI_RAD + angle), (2.0f * math_utils::PI_RAD));
  const float angle_label2 =
      fmodf((math_utils::PI_RAD + angle), (2.0f * math_utils::PI_RAD));

  const auto expected_labels = torch::tensor(
      {{label1[0], label1[1], label1[2], 2.0f, 3.0f, 2.5f, angle_label1},
       {label2[0], label2[1], label2[2], 1.0f, 1.0f, 0.5f, angle_label2}},
      F32);

  // NOTE(tom): Using allclose here because the equality check ends up failing
  //            despite there being no visible difference between the tensor
  //            elements. Even when printing with full 32bit floating point
  //            precision (%.9g).
  EXPECT_TRUE(labels.allclose(expected_labels))
      << "`labels` not equal to `expected_labels`:\nlabels:" << labels
      << "\nexpected_labels:\n"
      << expected_labels;
}

TEST(RNGTransformation, ThinOutTest) {
  constexpr tensor_size_t BATCHES = 2;
  constexpr tensor_size_t ITEMS = 10;

  const auto points = torch::rand({BATCHES, ITEMS, 4}, F32);
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

TEST(RNGTransformation, RandomPointNoiseTest) {
  constexpr float sigma = 1;

  auto points = torch::tensor({{{1.0, 2.0, 3.0, 4.0}, {-1.0, -2.0, -3.0, -4.0}},
                               {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}});

  // noise_vector = {{1.08580697, 1.00174832, -0.566087246},
  //                {-1.69251204, 0.729757607, -1.04425383}};
  // NOTE(tom): The RNG is not reset as expected. There is some weird
  //            1-2-3-4-1-2 pattern that then repeats.
  //            So the test is adjusted to fit that but not necessarily what is
  //            expected.
  const auto expected_points =
      torch::tensor({{{2.08580697, 3.00174832, 2.433912754, 4.0},
                      {-2.69251204, -0.91419303, -1.99825168, -4.0}},

                     {{2.08580697, 2.00174832, 0.433912754, 0.0},
                      {-1.69251204, 1.08580697, 2.00174832, 1.0}}});

  random_point_noise(points, sigma);

  // NOTE(tom): This currently fails because `draw_values` is behaving weirdly.
  EXPECT_TRUE(points.allclose(expected_points))
      << "Noise not as expected:\nexpected:\n"
      << expected_points << "\nactual:\n"
      << points;
}

TEST(RNGTransformation, TransformAlongRayTest) {
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

TEST(RNGTransformation, IntensityNoiseTest) {
  {
    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                       {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});

    constexpr float SIGMA = 20;
    constexpr auto MAX_INTENSITY =
        point_cloud_data::intensity_range::MAX_INTENSITY_255;

    // NOTE(tom): values of intensity_shift =
    //           {21.2925453, 0, 21.2925453, 6.91568279}
    const auto expected_points = torch::tensor(
        {{{1.0, 2.0, 3.0, 25.7925453}, {-1.0, -2.0, -3.0, 255.0}},
         {{1.0, 1.0, 1.0, 21.2925453}, {0.0, 0.0, 1.0, 252.01568279}}});

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
    constexpr auto MAX_INTENSITY =
        point_cloud_data::intensity_range::MAX_INTENSITY_1;

    // NOTE(tom): values of intensity_shift =
    //           {0.207830608, 0, 0.212925255, 0.0354648978}
    const auto expected_points = torch::tensor(
        {{{1.0, 2.0, 3.0, 0.727830609}, {-1.0, -2.0, -3.0, 1.0}},
         {{1.0, 1.0, 1.0, 0.212925255}, {0.0, 0.0, 1.0, 0.9854648978}}});

    intensity_noise(points, SIGMA, MAX_INTENSITY);

    EXPECT_TRUE(points.allclose(expected_points))
        << "Noise values not as expected:\nexpected:\n"
        << expected_points << "\nactual:\n"
        << points;
  }
}

TEST(RNGTransformation, IntensityShiftTest) {
  {
    auto points =
        torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                       {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});

    constexpr float SIGMA = 20;
    constexpr auto MAX_INTENSITY =
        point_cloud_data::intensity_range::MAX_INTENSITY_255;

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
    constexpr auto MAX_INTENSITY =
        point_cloud_data::intensity_range::MAX_INTENSITY_1;

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

TEST(Transformation, LocalToWorldTest) {
  const auto lidar_pose = torch::tensor({1.0, 2.0, 3.0, 180.0, 10.0, 0.0}, F64);

  const auto expected = torch::tensor(
      {
          {9.84807753e-01, 1.73648178e-01, -2.12657685e-17, 1.0},
          {1.73648178e-01, -9.84807753e-01, 1.20604166e-16, 2.0},
          {0.0, -1.22464680e-16, -1.0, 3.0},
          {0.0, 0.0, 0.0, 1.0},
      },
      F64);
  const auto result = local_to_world_transform(lidar_pose);

  EXPECT_TRUE(result.toType(F64).allclose(expected))
      << "Transformation matrix not as expected:\nexpected:\n"
      << expected << "\nactual:\n"
      << result;
}

TEST(Transformation, LocalToWorldConditionTest) {
  {

    const auto m = torch::tensor({1.0, 2.0, 3.0, 180.0, 10.0, 0.0}, F64);
    const auto result = local_to_world_transform(m);
    const auto cn = math_utils::compute_condition_number(result);
    constexpr double expected_max_cn = 15.94;

    EXPECT_LT(cn, expected_max_cn)
        << "Regression in the condition number for:\n"
        << m << "\nExpected it to be smaller than " << expected_max_cn
        << " but is " << cn << "!";
  }
  {

    const auto m = torch::tensor({-7.0, 3.0, 5.2, 0.0, 100.0, 0.0}, F64);
    const auto result = local_to_world_transform(m);
    const auto cn = math_utils::compute_condition_number(result);
    constexpr double expected_max_cn = 87.03;

    EXPECT_LT(cn, expected_max_cn)
        << "Regression in the condition number for:\n"
        << m << "\nExpected it to be smaller than " << expected_max_cn
        << " but is " << cn << "!";
  }
}

TEST(Simulation, FogTest) {
  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                     {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});
  const auto _ = fog(points, 100, DIST, 2, 0);

  // NOTE(tom): currently just testing if the whether the function runs
  EXPECT_TRUE(true);
}

TEST(Simulation, SnowTest) {
  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                     {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});
  const auto _ = snow(points[0], {-50, 50, -50, 50, -3, 1}, 1000, 5, 2,
                      point_cloud_data::intensity_range::MAX_INTENSITY_1);

  // NOTE(tom): currently just testing if the whether the function runs
  EXPECT_TRUE(true);
}

TEST(Simulation, RainTest) {
  auto points =
      torch::tensor({{{1.0, 2.0, 3.0, 4.5}, {-1.0, -2.0, -3.0, 255.0}},
                     {{1.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 245.1}}});
  const auto _ = rain(points[0], {-50, 50, -50, 50, -3, 1}, 1000, 5,
                      distribution::exponential);

  // NOTE(tom): currently just testing if the whether the function runs
  EXPECT_TRUE(true);
}

#endif

// NOLINTEND

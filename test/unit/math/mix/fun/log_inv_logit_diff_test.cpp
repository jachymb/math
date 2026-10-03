#include <test/unit/math/test_ad.hpp>
#include <limits>
#include <cmath>
#include <utility>
#include <vector>

TEST(mathMixScalFun, logInvLogitDiff) {
  auto f = [](const auto& x1, const auto& x2) {
    return stan::math::log_inv_logit_diff(x1, x2);
  };
  stan::test::expect_ad(f, 0.5, -1.0);
  stan::test::expect_ad(f, 0.5, 0.0);
  stan::test::expect_ad(f, 1.2, 0.6);
  stan::test::expect_ad(f, 3.4, 0.9);

  double nan = std::numeric_limits<double>::quiet_NaN();
  stan::test::expect_ad(f, 1.0, nan);
  stan::test::expect_ad(f, nan, 1.0);
  stan::test::expect_ad(f, nan, nan);
}

TEST(mathMixScalFun, logInvLogitDiff_vec) {
  auto f = [](const auto& x1, const auto& x2) {
    using stan::math::log_inv_logit_diff;
    return log_inv_logit_diff(x1, x2);
  };

  Eigen::VectorXd in1(2);
  in1 << 3, 1;
  Eigen::VectorXd in2(2);
  in2 << 0.5, -3.4;
  stan::test::expect_ad_vectorized_binary(f, in1, in2);
}

TEST(mathMixScalFun, logInvLogitDiffGradientLargeArgs) {
  using stan::math::fvar;
  using stan::math::inv_logit;
  using stan::math::log_inv_logit_diff;
  using stan::math::var;
  // d/dx = inv_logit(-x) + 1 / expm1(x - y), d/dy = -inv_logit(y) - 1 /
  // expm1(x - y): sums of positive terms, accurate references
  std::vector<std::pair<double, double>> xy{{40, 0},   {100, 1},   {30, -30},
                                            {37, 36},  {-30, -40}, {1, 0},
                                            {700, 20}, {0.5, -1.0}};
  for (auto p : xy) {
    double x = p.first;
    double y = p.second;
    double c = 1 / std::expm1(x - y);
    double dx = inv_logit(-x) + c;
    double dy = -inv_logit(y) - c;

    var xv = x;
    var yv = y;
    log_inv_logit_diff(xv, yv).grad();
    EXPECT_NEAR(xv.adj(), dx, 1e-13 * dx) << x << ", " << y;
    EXPECT_NEAR(yv.adj(), dy, -1e-13 * dy) << x << ", " << y;
    stan::math::recover_memory();

    var xv2 = x;
    log_inv_logit_diff(xv2, y).grad();
    EXPECT_NEAR(xv2.adj(), dx, 1e-13 * dx) << x << ", " << y;
    stan::math::recover_memory();

    EXPECT_NEAR(log_inv_logit_diff(fvar<double>(x, 1), y).d_, dx, 1e-13 * dx)
        << x << ", " << y;
    EXPECT_NEAR(log_inv_logit_diff(fvar<double>(x, 1), fvar<double>(y, 0)).d_,
                dx, 1e-13 * dx)
        << x << ", " << y;
    EXPECT_NEAR(log_inv_logit_diff(fvar<double>(x, 0), fvar<double>(y, 1)).d_,
                dy, -1e-13 * dy)
        << x << ", " << y;
  }
}

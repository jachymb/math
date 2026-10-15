#include <stan/math/mix.hpp>
#include <test/unit/math/rev/util.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

TEST_F(AgradRev, mathMixScalFun_poisson_log_lpmf_zero_rate) {
  // log p(0 | alpha) = -exp(alpha), which is 0 with all derivatives 0 at
  // alpha = -inf
  double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(0, stan::math::poisson_log_lpmf(0, -inf));
  auto f = [](const auto& x) { return stan::math::poisson_log_lpmf(0, x(0)); };
  Eigen::VectorXd x(1);
  x << -inf;
  double fx;
  Eigen::VectorXd grad;
  Eigen::MatrixXd H;
  stan::math::gradient(f, x, fx, grad);
  EXPECT_EQ(0, fx);
  EXPECT_EQ(0, grad(0));
  stan::math::hessian(f, x, fx, grad, H);
  EXPECT_EQ(0, fx);
  EXPECT_EQ(0, grad(0));
  EXPECT_EQ(0, H(0, 0));
  stan::math::hessian<double>(f, x, fx, grad, H);
  EXPECT_EQ(0, fx);
  EXPECT_EQ(0, grad(0));
  EXPECT_EQ(0, H(0, 0));

  // log p(n | alpha) = n alpha - exp(alpha) - log(n!)
  auto g = [](const auto& x) {
    std::vector<int> n{0, 2};
    return stan::math::poisson_log_lpmf(n, x)
           + stan::math::poisson_log_lpmf(0, x)
           + stan::math::poisson_log_lpmf(std::vector<int>{0, 0}, x(0));
  };
  Eigen::VectorXd y(2);
  y << -inf, 0;
  stan::math::gradient(g, y, fx, grad);
  EXPECT_FLOAT_EQ(-2 - std::log(2.0), fx);
  EXPECT_FLOAT_EQ(0, grad(0));
  EXPECT_FLOAT_EQ(0, grad(1));
}

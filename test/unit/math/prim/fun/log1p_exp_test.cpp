#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/next.hpp>
#include <cmath>
#include <limits>
#include <vector>

TEST(MathFunctions, log1p_exp) {
  using stan::math::log1p_exp;

  // exp(10000.0) overflows
  EXPECT_FLOAT_EQ(10000.0, log1p_exp(10000.0));
  EXPECT_FLOAT_EQ(0.0, log1p_exp(-10000.0));
}

TEST(MathFunctions, log1p_exp_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(std::isnan(stan::math::log1p_exp(nan)));
}

namespace log1p_exp_test {
// Container result c at x matches the scalar log1p_exp(x) within 4 ulp
void expect_matches_scalar(double x, double c) {
  double s = stan::math::log1p_exp(x);
  if (std::isnan(s)) {
    EXPECT_TRUE(std::isnan(c)) << "x = " << x;
  } else if (std::isinf(s) || s == 0) {
    EXPECT_EQ(s, c) << "x = " << x;
  } else {
    EXPECT_LE(std::abs(boost::math::float_distance(s, c)), 4) << "x = " << x;
  }
}
}  // namespace log1p_exp_test

TEST(MathFunctions, log1p_exp_containers_match_scalar) {
  using log1p_exp_test::expect_matches_scalar;
  using stan::math::log1p_exp;
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> xs{inf,  -inf, nan,   -1000, -746, -745,    -740,
                         -709, -3,   -0.45, -0.0,  0.0,  -1e-300, 1e-300,
                         2,    37,   709,   710,   1e308};
  const int n = xs.size();

  std::vector<double> r_std = log1p_exp(xs);
  ASSERT_EQ(n, r_std.size());
  for (int i = 0; i < n; ++i) {
    expect_matches_scalar(xs[i], r_std[i]);
  }

  std::vector<int> xi{-1000, -746, -745, -740, -709, -3, 0, 2, 37, 709, 710};
  std::vector<double> r_int = log1p_exp(xi);
  ASSERT_EQ(xi.size(), r_int.size());
  for (size_t i = 0; i < xi.size(); ++i) {
    expect_matches_scalar(xi[i], r_int[i]);
  }

  Eigen::VectorXd v = Eigen::Map<Eigen::VectorXd>(xs.data(), n);
  Eigen::VectorXd r_v = log1p_exp(v);
  for (int i = 0; i < n; ++i) {
    expect_matches_scalar(v[i], r_v[i]);
  }

  Eigen::RowVectorXd rv = v.transpose();
  Eigen::RowVectorXd r_rv = log1p_exp(rv);
  for (int i = 0; i < n; ++i) {
    expect_matches_scalar(rv[i], r_rv[i]);
  }

  Eigen::MatrixXd m2(2, n);
  m2.row(0) = rv;
  m2.row(1) = -rv;
  Eigen::MatrixXd r_m = log1p_exp(m2);
  ASSERT_EQ(2, r_m.rows());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < n; ++j) {
      expect_matches_scalar(m2(i, j), r_m(i, j));
    }
  }

  std::vector<Eigen::VectorXd> vv{v, -v};
  std::vector<Eigen::VectorXd> r_vv = log1p_exp(vv);
  ASSERT_EQ(2, r_vv.size());
  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < n; ++i) {
      expect_matches_scalar(vv[k][i], r_vv[k][i]);
    }
  }

  Eigen::VectorXd r_expr = log1p_exp(v.reverse() * 1.0);
  for (int i = 0; i < n; ++i) {
    expect_matches_scalar(v[n - 1 - i], r_expr[i]);
  }
}

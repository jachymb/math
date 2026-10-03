#include <stan/math/rev.hpp>
#include <test/unit/math/test_ad.hpp>
#include <gtest/gtest.h>
#include <boost/math/differentiation/finite_difference.hpp>

TEST_F(AgradRev, mathMixScalFun_neg_binomial_2_log_lpmf_derivatives) {
  auto f1 = [](const auto& eta, const auto& phi) {
    return stan::math::neg_binomial_2_log_lpmf(0, eta, phi);
  };
  auto f2 = [](const auto& eta, const auto& phi) {
    return stan::math::neg_binomial_2_log_lpmf(6, eta, phi);
  };

  stan::test::expect_ad(f1, -1.5, 4.1);
  stan::test::expect_ad(f1, 2.0, 1.1);
  stan::test::expect_ad(f2, -1.5, 4.1);
  stan::test::expect_ad(f2, 2.0, 1.1);
}

// Hessian where exp(eta) / phi is far beyond 1 / epsilon (eta = 35), where
// exp(eta) overflows (720) or underflows (-800), and below log(phi) (-3).
// Partials written in exp(eta) are NaN in the middle two.
TEST_F(AgradRev, mathMixScalFun_neg_binomial_2_log_lpmf_hessian_extreme) {
  const double phi = 0.07;
  for (int n : {0, 3}) {
    for (double eta : {35.0, 720.0, -800.0, -3.0}) {
      // Closed form via s = inv_logit(eta - log(phi)).
      const double u = eta - std::log(phi);
      const double e = std::exp(-std::abs(u));
      const double s = u > 0 ? 1 / (1 + e) : e / (1 + e);
      const double t = u > 0 ? e / (1 + e) : 1 / (1 + e);  // 1 - s
      Eigen::Matrix2d H_expected, tol;
      H_expected << -(n + phi) * s * t, -s + (n + phi) / phi * s * t, 0,
          s * s / phi + n * t * t / (phi * phi) + stan::math::trigamma(n + phi)
              - stan::math::trigamma(phi);
      H_expected(1, 0) = H_expected(0, 1);
      tol << 1e-10 * std::abs(H_expected(0, 0)),
          1e-10 * (s + (n + phi) / phi * s * t), 0,
          1e-10
              * (s * s / phi + n * t * t / (phi * phi)
                 + stan::math::trigamma(phi));
      tol(1, 0) = tol(0, 1);

      auto f = [n](const auto& v) {
        return stan::math::neg_binomial_2_log_lpmf(n, v(0), v(1));
      };
      Eigen::VectorXd v(2);
      v << eta, phi;
      double fx;
      Eigen::VectorXd grad;
      Eigen::MatrixXd H_mix, H_fwd;
      stan::math::hessian(f, v, fx, grad, H_mix);
      stan::math::hessian<double>(f, v, fx, grad, H_fwd);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          EXPECT_NEAR(H_mix(i, j), H_expected(i, j), tol(i, j))
              << "fvar<var> (" << i << ", " << j << "), n " << n << ", eta "
              << eta;
          EXPECT_NEAR(H_fwd(i, j), H_expected(i, j), tol(i, j))
              << "fvar<fvar<double>> (" << i << ", " << j << "), n " << n
              << ", eta " << eta;
        }
      }
    }
  }
}

#include <stan/math/mix.hpp>
#include <test/unit/math/test_ad.hpp>

TEST_F(AgradRev, mathMixScalFun_neg_binomial_2_log_glm_lpmf) {
  auto f = [](const auto y, const auto& x) {
    return [=](const auto& alpha, const auto& beta, const auto& phi) {
      return stan::math::neg_binomial_2_log_glm_lpmf(y, x, alpha, beta, phi);
    };
  };
  auto f2 = [](const auto y, const auto& phi) {
    return [=](const auto& x, const auto& alpha, const auto& beta) {
      return stan::math::neg_binomial_2_log_glm_lpmf(y, x, alpha, beta, phi);
    };
  };

  std::vector<int> y{0, 1};
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(2, 2);
  Eigen::RowVectorXd x_rowvec = x.row(0);
  Eigen::VectorXd alpha = Eigen::VectorXd::Random(2);
  Eigen::VectorXd beta = Eigen::VectorXd::Random(2);
  double phi = 1.5;

  stan::test::expect_ad(f(y[0], x), alpha, beta, phi);
  stan::test::expect_ad(f(y[0], x), alpha[0], beta, phi);
  stan::test::expect_ad(f(y[0], x_rowvec), alpha, beta, phi);
  stan::test::expect_ad(f(y[0], x_rowvec), alpha[0], beta, phi);

  stan::test::expect_ad(f2(y, phi), x, alpha, beta);
  stan::test::expect_ad(f2(y, phi), x, alpha[0], beta);
  stan::test::expect_ad(f2(y, phi), x_rowvec, alpha, beta);
  stan::test::expect_ad(f2(y, phi), x_rowvec, alpha[0], beta);
}

// Second derivatives in (beta, phi) where the mean is far above the
// dispersion: exp(eta) / phi passes 1 / epsilon and, in the last row,
// exp(eta) overflows, as Newton iterates of the embedded Laplace approximation
// can. Every entry of the Hessian must be finite, and those involving beta
// accurate relative to their own size, which is tiny.
TEST_F(AgradRev, neg_binomial_2_log_glm_lpmf_hessian_large_mean) {
  std::vector<int> y{1, 3, 0, 2};
  Eigen::MatrixXd x(4, 2);
  x << 1, 0.5, 1, -0.3, 1, 0.9, 1, 69;
  Eigen::VectorXd beta_phi(3);
  beta_phi << 30, 10, 0.07;  // eta = 35, 27, 39, 720; phi = 0.07
  auto f = [&](const auto& v) {
    return stan::math::neg_binomial_2_log_glm_lpmf(y, x, 0.0, v.head(2),
                                                   v(2));
  };
  double fx;
  Eigen::VectorXd grad;
  Eigen::MatrixXd H;
  stan::math::hessian(f, beta_phi, fx, grad, H);
  EXPECT_TRUE(H.allFinite()) << H;

  // With s = inv_logit(eta - log(phi)), d log p / d eta = y - (y + phi) s,
  // so d2 / d eta2 = -(y + phi) s (1 - s) and
  // d2 / d eta d phi = -s + (y + phi) / phi s (1 - s).
  const double phi = beta_phi(2);
  Eigen::VectorXd eta = x * beta_phi.head(2);
  Eigen::VectorXd w(4), c(4);
  for (int i = 0; i < 4; ++i) {
    const double e = std::exp(-std::abs(eta(i) - std::log(phi)));
    const double s = eta(i) > std::log(phi) ? 1 / (1 + e) : e / (1 + e);
    const double s1ms = e / ((1 + e) * (1 + e));
    w(i) = (y[i] + phi) * s1ms;
    c(i) = -s + (y[i] + phi) / phi * s1ms;
  }
  Eigen::MatrixXd H_beta = -x.transpose() * w.asDiagonal() * x;
  Eigen::VectorXd H_beta_phi = x.transpose() * c;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_NEAR(H(i, j), H_beta(i, j), 1e-10 * std::abs(H_beta(i, j)))
          << "(" << i << ", " << j << ")";
    }
    EXPECT_NEAR(H(i, 2), H_beta_phi(i), 1e-10 * std::abs(H_beta_phi(i)))
        << "(" << i << ", phi)";
  }
}

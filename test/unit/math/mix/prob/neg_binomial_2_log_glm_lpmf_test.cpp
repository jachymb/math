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

namespace {
// Second derivatives of the NB2 log density in (eta, phi), via
// s = inv_logit(eta - log(phi)).
Eigen::Vector3d nb2_log_hessian(int n, double eta, double phi) {
  const double u = eta - std::log(phi);
  const double e = std::exp(-std::abs(u));
  const double s = u > 0 ? 1 / (1 + e) : e / (1 + e);
  const double t = u > 0 ? e / (1 + e) : 1 / (1 + e);  // 1 - s
  return {-(n + phi) * s * t, -s + (n + phi) / phi * s * t,
          s * s / phi + n * t * t / (phi * phi) + stan::math::trigamma(n + phi)
              - stan::math::trigamma(phi)};
}
}  // namespace

// Hessian where exp(eta) / phi is far beyond 1 / epsilon (eta = 35, 27), where
// exp(eta) overflows (720) or underflows (-773), and below log(phi) (-3).
// Partials written in exp(eta) keep at most 3 digits in the first two and are
// NaN in the next two.
TEST_F(AgradRev, mathMixScalFun_neg_binomial_2_log_glm_lpmf_hessian_extreme) {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  const std::vector<int> y{1, 3, 0, 2, 4};
  const VectorXd eta = (VectorXd(5) << 35, 27, -3, 720, -773).finished();
  const MatrixXd x = (MatrixXd(5, 1) << 0.5, -0.3, 2, 1, -1).finished();
  const double beta = 1;
  const double phi = 0.07;
  for (bool vector_phi : {false, true}) {
    // v = (alpha, beta, phi), eta = alpha + x * beta
    const int n_phi = vector_phi ? 5 : 1;
    VectorXd v(6 + n_phi);
    v << eta - x * beta, beta, VectorXd::Constant(n_phi, phi);
    auto f = [&](const auto& v) {
      const auto& alpha = v.head(5);
      if (vector_phi) {
        return stan::math::neg_binomial_2_log_glm_lpmf(
            y, x, alpha, v.segment(5, 1), v.tail(5));
      }
      return stan::math::neg_binomial_2_log_glm_lpmf(y, x, alpha,
                                                     v.segment(5, 1), v(6));
    };
    MatrixXd J_eta = MatrixXd::Zero(5, v.size());
    MatrixXd J_phi = MatrixXd::Zero(5, v.size());
    J_eta.leftCols(5).setIdentity();
    J_eta.col(5) = x;
    if (vector_phi) {
      J_phi.rightCols(5).setIdentity();
    } else {
      J_phi.col(6).setOnes();
    }
    MatrixXd H_expected = MatrixXd::Zero(v.size(), v.size());
    MatrixXd tol = H_expected;
    for (int i = 0; i < 5; ++i) {
      const Eigen::Vector3d h = nb2_log_hessian(y[i], eta(i), phi);
      Eigen::Matrix2d h_i;
      h_i << h(0), h(1), h(1), h(2);
      Eigen::MatrixXd J_i(2, v.size());
      J_i << J_eta.row(i), J_phi.row(i);
      H_expected += J_i.transpose() * h_i * J_i;
      tol += 1e-10 * J_i.cwiseAbs().transpose() * h_i.cwiseAbs()
             * J_i.cwiseAbs();
    }
    double fx;
    VectorXd grad;
    MatrixXd H_mix, H_fwd;
    stan::math::hessian(f, v, fx, grad, H_mix);
    stan::math::hessian<double>(f, v, fx, grad, H_fwd);
    for (int i = 0; i < v.size(); ++i) {
      for (int j = 0; j < v.size(); ++j) {
        EXPECT_NEAR(H_mix(i, j), H_expected(i, j), tol(i, j))
            << "fvar<var> (" << i << ", " << j << "), vector phi "
            << vector_phi;
        EXPECT_NEAR(H_fwd(i, j), H_expected(i, j), tol(i, j))
            << "fvar<fvar<double>> (" << i << ", " << j << "), vector phi "
            << vector_phi;
      }
    }
  }
}

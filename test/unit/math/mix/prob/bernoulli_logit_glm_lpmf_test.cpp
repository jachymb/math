#include <stan/math/mix.hpp>
#include <test/unit/math/test_ad.hpp>
#include <cmath>
#include <limits>
#include <vector>

TEST_F(AgradRev, mathMixScalFun_bernoulli_logit_glm_lpmf) {
  auto f = [](const auto y) {
    return [=](const auto& x, const auto& alpha, const auto& beta) {
      return stan::math::bernoulli_logit_glm_lpmf(y, x, alpha, beta);
    };
  };

  std::vector<int> y{0, 1};
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(2, 2);
  Eigen::RowVectorXd x_rowvec = x.row(0);
  Eigen::VectorXd alpha = Eigen::VectorXd::Random(2);
  Eigen::VectorXd beta = Eigen::VectorXd::Random(2);

  stan::test::expect_ad(f(y[0]), x, alpha, beta);
  stan::test::expect_ad(f(y[0]), x, alpha[0], beta);
  stan::test::expect_ad(f(y[1]), x_rowvec, alpha, beta);
  stan::test::expect_ad(f(y[1]), x_rowvec, alpha[0], beta);
  stan::test::expect_ad(f(y), x, alpha, beta);
  stan::test::expect_ad(f(y), x, alpha[0], beta);
  stan::test::expect_ad(f(y), x_rowvec, alpha, beta);
  stan::test::expect_ad(f(y), x_rowvec, alpha[0], beta);
}

namespace bernoulli_logit_glm_test {
// p(1 - p) at theta, without overflow
inline double logistic_weight(double theta) {
  double a = std::exp(-std::fabs(theta));
  return a / ((1 + a) * (1 + a));
}

// Hessian wrt (alpha, beta) of sum_n log p(y_n | theta_n),
// theta = alpha + x * beta, is -J' diag(p(1 - p)) J with J = [I, x].
// tol(i, j) is relative to sum_n p(1 - p)_n |J_ni| |J_nj|, plus the
// smallest normal double (p(1 - p) at |theta| = 720 is subnormal).
template <typename T_x>
void expect_hessian(const std::vector<int>& y, const T_x& x,
                    const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta) {
  using stan::math::fvar;
  const int N = y.size();
  const int K = beta.size();
  Eigen::MatrixXd J(N, N + K);
  J << Eigen::MatrixXd::Identity(N, N), x.replicate(N / x.rows(), 1);
  Eigen::VectorXd theta = J.rightCols(K) * beta + alpha;
  Eigen::VectorXd w = theta.unaryExpr(&logistic_weight);
  Eigen::MatrixXd H = -J.transpose() * w.asDiagonal() * J;
  Eigen::MatrixXd tol
      = (1e-13 * J.cwiseAbs().transpose() * w.asDiagonal() * J.cwiseAbs())
            .array()
        + std::numeric_limits<double>::min();

  auto f = [&](const auto& v) {
    return stan::math::bernoulli_logit_glm_lpmf(y, x, v.head(N), v.tail(K));
  };
  Eigen::VectorXd v(N + K);
  v << alpha, beta;

  double fx;
  Eigen::VectorXd grad;
  Eigen::MatrixXd H_rev;
  stan::math::hessian(f, v, fx, grad, H_rev);

  Eigen::MatrixXd H_fwd(N + K, N + K);
  for (int i = 0; i < N + K; ++i) {
    for (int j = 0; j < N + K; ++j) {
      Eigen::Matrix<fvar<fvar<double>>, Eigen::Dynamic, 1> v_ff(N + K);
      for (int k = 0; k < N + K; ++k) {
        v_ff(k) = fvar<fvar<double>>(fvar<double>(v(k), i == k), j == k);
      }
      H_fwd(i, j) = f(v_ff).d_.d_;
    }
  }

  for (int i = 0; i < N + K; ++i) {
    for (int j = 0; j < N + K; ++j) {
      EXPECT_NEAR(H_rev(i, j), H(i, j), tol(i, j)) << i << ", " << j;
      EXPECT_NEAR(H_fwd(i, j), H(i, j), tol(i, j)) << i << ", " << j;
    }
  }
}
}  // namespace bernoulli_logit_glm_test

// y * theta = -30 and -19.5: curvature p(1 - p) ~ exp(y * theta), not 0.
TEST_F(AgradRev, mathMixScalFun_bernoulli_logit_glm_lpmf_hessian_tails) {
  std::vector<int> y{1, 0, 1, 0, 1, 0};
  Eigen::MatrixXd x(6, 2);
  x << 0.5, -1, 1, 0.25, -0.5, 2, 1.5, -0.75, 0.2, 0.1, -1, -0.3;
  Eigen::VectorXd beta(2);
  beta << 0.4, -0.2;
  Eigen::VectorXd theta(6);
  theta << -30, 30, -19.5, 19.5, 0.25, -0.3;
  Eigen::VectorXd alpha = theta - x * beta;
  bernoulli_logit_glm_test::expect_hessian(y, x, alpha, beta);

  theta << 30, -30, 19.5, -19.5, 35, -40;
  alpha = theta - x * beta;
  bernoulli_logit_glm_test::expect_hessian(y, x, alpha, beta);
}

// y * theta = -720: exp(-y * theta) overflows; the Hessian must not be NaN.
TEST_F(AgradRev, mathMixScalFun_bernoulli_logit_glm_lpmf_hessian_overflow) {
  std::vector<int> y{1, 0, 1, 0};
  Eigen::MatrixXd x(4, 2);
  x << 0.5, -1, 1, 0.25, -0.5, 2, 1.5, -0.75;
  Eigen::VectorXd beta(2);
  beta << 0.4, -0.2;
  Eigen::VectorXd theta(4);
  theta << -720, 720, 0.5, -2500;
  Eigen::VectorXd alpha = theta - x * beta;
  bernoulli_logit_glm_test::expect_hessian(y, x, alpha, beta);

  Eigen::RowVectorXd x_row = x.row(0);
  alpha = theta.array() - x_row.dot(beta);
  bernoulli_logit_glm_test::expect_hessian(y, x_row, alpha, beta);
}

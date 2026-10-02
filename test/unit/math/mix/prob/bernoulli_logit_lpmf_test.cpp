#include <stan/math/mix.hpp>
#include <test/unit/math/test_ad.hpp>
#include <cmath>
#include <limits>
#include <vector>

// y * theta in {-2500, -720, -30, -19.5, 0.25, 19.5, 30}: Hessian is
// -diag(p(1 - p)): finite, and not 0 in the tails.
TEST_F(AgradRev, mathMixScalFun_bernoulli_logit_lpmf_hessian_tails) {
  using stan::math::fvar;
  std::vector<int> y{1, 0, 1, 0, 1, 0, 1};
  Eigen::VectorXd theta(7);
  theta << -720, 30, -19.5, -0.25, 19.5, -30, -2500;
  const int N = theta.size();
  auto f
      = [&](const auto& v) { return stan::math::bernoulli_logit_lpmf(y, v); };

  double fx;
  Eigen::VectorXd grad;
  Eigen::MatrixXd H_rev;
  stan::math::hessian(f, theta, fx, grad, H_rev);

  for (int i = 0; i < N; ++i) {
    double a = std::exp(-std::fabs(theta(i)));
    double w = a / ((1 + a) * (1 + a));
    double tol = 1e-13 * w + std::numeric_limits<double>::min();
    Eigen::Matrix<fvar<fvar<double>>, Eigen::Dynamic, 1> v_ff(N);
    for (int k = 0; k < N; ++k) {
      v_ff(k) = fvar<fvar<double>>(fvar<double>(theta(k), i == k), i == k);
    }
    double h_fwd = f(v_ff).d_.d_;
    for (int j = 0; j < N; ++j) {
      EXPECT_NEAR(H_rev(i, j), i == j ? -w : 0, tol) << i << ", " << j;
    }
    EXPECT_NEAR(h_fwd, -w, tol) << i;
  }
}

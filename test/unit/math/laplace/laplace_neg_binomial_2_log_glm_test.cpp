#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <tuple>
#include <vector>

namespace {

// A hierarchical negative binomial regression for one group: the latent
// vector is the group's deviation from the population coefficients mu.
struct NegBinomial2LogGlmLikelihood {
  template <typename Theta, typename Mu, typename Phi>
  auto operator()(const Theta& theta, const std::vector<int>& y,
                  const Eigen::MatrixXd& x, const Mu& mu, const Phi& phi,
                  std::ostream* /*msgs*/) const {
    return stan::math::neg_binomial_2_log_glm_lpmf(
        y, x, 0.0, stan::math::add(mu, theta), phi);
  }
};

struct DiagonalCovariance {
  template <typename Tau>
  auto operator()(const Tau& tau, std::ostream* /*msgs*/) const {
    return stan::math::diag_matrix(stan::math::square(tau));
  }
};

// Two observations, six coefficients, exp(eta) / phi above 1e9: W is positive
// semi-definite of rank 2. Without accurate second derivatives its zero
// eigenvalues come out negative and the Hessian-root solver (1) rejects W.
struct Problem {
  std::vector<int> y{1, 3};
  Eigen::MatrixXd x;
  Eigen::VectorXd mu;
  Eigen::VectorXd tau;
  double phi = 0.07;
  Problem() : x(2, 6), mu(6), tau(6) {
    x << 1, 0.9, -0.4, 0.3, 0.8, -0.6, 1, 0.7, 0.2, -0.5, 0.4, 0.9;
    mu << 19.0, 0.5, -0.3, 0.2, 0.1, -0.2;
    tau << 1.0, 0.5, 0.5, 0.3, 0.3, 0.2;
  }
  double run(int solver) const {
    return stan::math::laplace_marginal_tol<false>(
        NegBinomial2LogGlmLikelihood{}, std::forward_as_tuple(y, x, mu, phi), 6,
        DiagonalCovariance{}, std::forward_as_tuple(tau),
        std::make_tuple(Eigen::VectorXd::Zero(6).eval(), 1e-10, 100, solver,
                        100, 0),
        nullptr);
  }
};

}  // namespace

TEST(LaplaceNegBinomial2LogGlm, BlockSolverAcceptsSingularHessianAtLargeMean) {
  const Problem p;
  double solver1 = 0;
  EXPECT_NO_THROW(solver1 = p.run(1));
  const double solver2 = p.run(2);
  EXPECT_TRUE(std::isfinite(solver1));
  EXPECT_NEAR(solver1, solver2, 1e-8 * std::abs(solver2));
}

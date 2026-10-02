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

// Two observations, six coefficients and a mean far above the dispersion
// (exp(eta) / phi near 1e10): W = X' diag(w) X is positive semi-definite of
// rank 2, with w near 1e-9. The second derivative of the GLM must be accurate
// relative to w, or W comes out indefinite at the scale of its largest
// eigenvalue and the block Hessian-root solver rejects it.
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
  template <typename Mu, typename Phi>
  auto run(const Mu& mu_, const Phi& phi_, int solver) const {
    return stan::math::laplace_marginal_tol<false>(
        NegBinomial2LogGlmLikelihood{}, std::forward_as_tuple(y, x, mu_, phi_),
        6, DiagonalCovariance{}, std::forward_as_tuple(tau),
        std::make_tuple(Eigen::VectorXd::Zero(6).eval(), 1e-10, 100, solver,
                        100, 0),
        nullptr);
  }
};

}  // namespace

TEST(LaplaceNegBinomial2LogGlm, BlockSolverAcceptsSingularHessianAtLargeMean) {
  const Problem p;
  double solver1 = 0;
  EXPECT_NO_THROW(solver1 = p.run(p.mu, p.phi, 1));
  const double solver2 = p.run(p.mu, p.phi, 2);
  EXPECT_TRUE(std::isfinite(solver1));
  EXPECT_NEAR(solver1, solver2, 1e-8 * std::abs(solver2));
}

TEST(LaplaceNegBinomial2LogGlm, GradientAtLargeMeanMatchesFiniteDifferences) {
  using stan::math::var;
  const Problem p;
  for (int solver : {1, 2}) {
    Eigen::Matrix<var, Eigen::Dynamic, 1> mu_v = p.mu;
    var phi_v = p.phi;
    var lp = p.run(mu_v, phi_v, solver);
    stan::math::grad(lp.vi_);
    Eigen::VectorXd mu_grad = mu_v.adj();
    const double phi_grad = phi_v.adj();
    stan::math::recover_memory();

    const double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
      Eigen::VectorXd up = p.mu, down = p.mu;
      up(k) += h;
      down(k) -= h;
      const double fd = (p.run(up, p.phi, solver) - p.run(down, p.phi, solver))
                        / (2 * h);
      EXPECT_NEAR(mu_grad(k), fd, 1e-5 * (1 + std::abs(fd)))
          << "solver " << solver << " mu[" << k << "]";
    }
    const double hp = h * p.phi;
    const double fd_phi
        = (p.run(p.mu, p.phi + hp, solver) - p.run(p.mu, p.phi - hp, solver))
          / (2 * hp);
    EXPECT_NEAR(phi_grad, fd_phi, 1e-5 * (1 + std::abs(fd_phi)))
        << "solver " << solver;
  }
}

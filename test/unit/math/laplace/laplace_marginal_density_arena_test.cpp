#include <stan/math.hpp>
#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace laplace_marginal_density_arena_test {

std::uintptr_t probe_base = 0;
std::intptr_t probe_peak = 0;

// Poisson log likelihood that records the highest arena position reached
// at its calls, relative to probe_base
struct probing_poisson {
  template <typename Theta>
  auto operator()(const Theta& theta, const std::vector<int>& y,
                  std::ostream* msgs) const {
    if (probe_base != 0) {
      auto position = reinterpret_cast<std::uintptr_t>(
          stan::math::ChainableStack::instance_->memalloc_.alloc(8));
      probe_peak = std::max(probe_peak,
                            static_cast<std::intptr_t>(position - probe_base));
    }
    return stan::math::poisson_log_lpmf(y, theta);
  }
};

struct se_covariance {
  template <typename T1, typename T2>
  auto operator()(const std::vector<Eigen::VectorXd>& x, const T1& alpha,
                  const T2& rho, std::ostream* msgs) const {
    return stan::math::add_diag(stan::math::gp_exp_quad_cov(x, alpha, rho),
                                1e-6);
  }
};

// Peak arena bytes of one gradient with the given solver, with all
// allocations in one large arena block
std::intptr_t peak_arena_bytes(int solver, int n) {
  using stan::math::var;
  auto& memalloc = stan::math::ChainableStack::instance_->memalloc_;
  stan::math::recover_memory();
  memalloc.alloc(16 << 20);
  stan::math::recover_memory();
  memalloc.alloc(1 << 20);  // skip the small first block
  std::vector<Eigen::VectorXd> x(n, Eigen::VectorXd(2));
  std::vector<int> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] << std::sin(0.37 * i), std::cos(0.11 * i);
    y[i] = i % 4;
  }
  Eigen::VectorXd theta_0 = Eigen::VectorXd::Zero(n);
  var alpha = 1.2;
  var rho = 0.7;
  probe_base = reinterpret_cast<std::uintptr_t>(memalloc.alloc(8));
  probe_peak = 0;
  var lp = stan::math::laplace_marginal_tol<false>(
      probing_poisson{}, std::forward_as_tuple(y), 1, se_covariance{},
      std::forward_as_tuple(x, alpha, rho),
      std::make_tuple(theta_0, 1e-8, 200, solver, 0, true), nullptr);
  probe_base = 0;
  stan::math::recover_memory();
  return probe_peak;
}

}  // namespace laplace_marginal_density_arena_test

TEST(laplace_marginal_density, solvers_2_and_3_allocate_no_unused_buffers) {
  using laplace_marginal_density_arena_test::peak_arena_bytes;
  for (int n : {20, 50}) {
    std::intptr_t peak_1 = peak_arena_bytes(1, n);
    ASSERT_GT(peak_1, 0);
    ASSERT_LT(peak_1, 8 << 20);  // all probes in the same block
    for (int solver : {2, 3}) {
      // The solvers' arena use differs by much less than one n x n matrix;
      // an unused n x n buffer adds 8 n^2 bytes.
      EXPECT_LT(peak_arena_bytes(solver, n) - peak_1, 4 * n * n)
          << "solver " << solver << ", n = " << n;
    }
  }
}

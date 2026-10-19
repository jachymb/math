#include <stan/math/prim.hpp>
#include <test/unit/math/prim/prob/util.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <algorithm>
#include <complex>
#include <limits>
#include <vector>

namespace complex_uniform_test {
template <typename F>
void expect_domain_errors(const F& f) {
  using c = std::complex<double>;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(f(c(nan, 0), c(0, 0), 1.0), std::domain_error);
  EXPECT_THROW(f(c(0, nan), c(0, 0), 1.0), std::domain_error);
  EXPECT_THROW(f(c(0, 0), c(inf, 0), 1.0), std::domain_error);
  EXPECT_THROW(f(c(0, 0), c(0, nan), 1.0), std::domain_error);
  EXPECT_THROW(f(c(0, 0), c(0, 0), 0.0), std::domain_error);
  EXPECT_THROW(f(c(0, 0), c(0, 0), inf), std::domain_error);
}
}  // namespace complex_uniform_test

TEST(ProbDistributionsComplexUniform, lpdf) {
  using c = std::complex<double>;
  using stan::math::complex_uniform_lpdf;
  const c mu(1, -2);
  const double log_density = -std::log(4 * stan::math::pi());
  EXPECT_FLOAT_EQ(log_density, complex_uniform_lpdf(c(2, -1), mu, 2.0));
  EXPECT_FLOAT_EQ(log_density, complex_uniform_lpdf(c(1, 0), mu, 2.0));
  EXPECT_EQ(stan::math::LOG_ZERO, complex_uniform_lpdf(c(2.5, -0.5), mu, 2));
  EXPECT_EQ(0, complex_uniform_lpdf<true>(c(2, -1), mu, 2.0));
}

TEST(ProbDistributionsComplexUniform, lcdf) {
  using c = std::complex<double>;
  using stan::math::complex_uniform_lcdf;
  // {Re y, Im y, Re mu, Im mu, rho, lcdf}: log of the disk area in the
  // quadrant over pi, by mpmath quadrature of the slice lengths (50 digits)
  const std::vector<std::vector<double>> cases
      = {{0, 0, 0, 0, 1, -1.3862943611198906188},
         {0.5, -0.5, 0, 0, 1, -1.7694744640883963752},
         {0.8, 0.8, 0, 0, 1, -0.10991312826986028494},
         {-0.8, 0.9, 0, 0, 1, -2.955665392906922425},
         {0.9, -0.8, 0, 0, 1, -2.955665392906922425},
         {2.5, 1.0, 1, -2, 3, -0.21753569164062563638},
         {-0.5, 1.0, 0.5, 1.5, 2, -2.8336355139390582294},
         {1.5, 0.2, 0, 0, 1, -0.4676544532054735516},
         {1, 0, 0, 0, 1, -0.69314718055994530942}};
  for (const auto& r : cases) {
    EXPECT_NEAR(r[5], complex_uniform_lcdf(c(r[0], r[1]), c(r[2], r[3]), r[4]),
                1e-14);
  }
  EXPECT_EQ(0, complex_uniform_lcdf(c(1, 1), c(0, 0), 1));
  const double neg_inf = stan::math::NEGATIVE_INFTY;
  EXPECT_EQ(neg_inf, complex_uniform_lcdf(c(-1.5, 0.3), c(0, 0), 1));
  EXPECT_EQ(neg_inf, complex_uniform_lcdf(c(0.3, -1), c(0, 0), 1));
  EXPECT_EQ(neg_inf, complex_uniform_lcdf(c(-0.8, -0.8), c(0, 0), 1));
}

TEST(ProbDistributionsComplexUniform, errors) {
  using c = std::complex<double>;
  using stan::math::complex_uniform_rng;
  complex_uniform_test::expect_domain_errors([](auto y, auto mu, auto rho) {
    return stan::math::complex_uniform_lpdf(y, mu, rho);
  });
  complex_uniform_test::expect_domain_errors([](auto y, auto mu, auto rho) {
    return stan::math::complex_uniform_lcdf(y, mu, rho);
  });
  boost::random::mt19937 rng;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(complex_uniform_rng(c(inf, 0), 1.0, rng), std::domain_error);
  EXPECT_THROW(complex_uniform_rng(c(0, nan), 1.0, rng), std::domain_error);
  EXPECT_THROW(complex_uniform_rng(c(0, 0), -1.0, rng), std::domain_error);
  EXPECT_THROW(complex_uniform_rng(c(0, 0), inf, rng), std::domain_error);
}

TEST(ProbDistributionsComplexUniform, rng) {
  boost::random::mt19937 rng;
  const std::complex<double> mu(1, -2);
  const double rho = 3;
  const int N = 10000;
  const int K = 10;
  // uniform on the disk: |z - mu|^2 / rho^2 and arg(z - mu) / (2 pi) + 1 / 2
  // are independent standard uniforms, so each cell of a K x K grid gets 1 /
  // K^2
  std::vector<int> counts(K * K, 0);
  for (int n = 0; n < N; ++n) {
    const std::complex<double> d
        = stan::math::complex_uniform_rng(mu, rho, rng) - mu;
    const int i
        = std::min(K - 1, static_cast<int>(K * std::norm(d) / (rho * rho)));
    const int j = std::min(
        K - 1, static_cast<int>(K * (std::arg(d) / stan::math::TWO_PI + 0.5)));
    ++counts[i * K + j];
  }
  assert_chi_squared(counts, std::vector<double>(K * K, N / (K * K)), 1e-6);
}

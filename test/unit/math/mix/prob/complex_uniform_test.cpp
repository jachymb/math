#include <test/unit/math/test_ad.hpp>
#include <complex>
#include <limits>
#include <vector>

TEST_F(AgradRev, mathMixScalFun_complex_uniform_lpdf) {
  auto f = [](const auto& y, const auto& mu, const auto& rho) {
    return stan::math::complex_uniform_lpdf(y, mu, rho);
  };
  stan::test::expect_ad(f, std::complex<double>(0.3, -0.2),
                        std::complex<double>(0.1, 0.4), 1.5);
}

TEST_F(AgradRev, mathMixScalFun_complex_uniform_lpdf_propto) {
  using stan::math::var;
  const std::complex<var> y(0.3, -0.2), mu(0.1, 0.4);
  var rho = 1.5;
  var lp = stan::math::complex_uniform_lpdf<true>(y, mu, rho);
  EXPECT_FLOAT_EQ(-2 * std::log(1.5), lp.val());
  lp.grad();
  EXPECT_FLOAT_EQ(-2 / 1.5, rho.adj());
  EXPECT_EQ(0, stan::math::complex_uniform_lpdf<true>(y, mu, 1.5).val());
}

TEST_F(AgradRev, mathMixScalFun_complex_uniform_lcdf) {
  auto f = [](const auto& y, const auto& mu, const auto& rho) {
    return stan::math::complex_uniform_lcdf(y, mu, rho);
  };
  const std::complex<double> mu(0.5, -1);
  const double rho = 2;
  // standardized corners (y - mu) / rho in each region of the cdf: inside the
  // disk, beyond the arc in the first quadrant, only one cut binding, F = 1
  const std::vector<std::complex<double>> corners
      = {{0.5, -0.5}, {-0.3, 0.2}, {-0.3, -0.4}, {0.8, 0.8}, {-0.8, 0.9},
         {0.9, -0.8}, {1.5, 0.3},  {0.3, 1.5},   {1.5, 1.5}};
  for (const auto& uv : corners) {
    stan::test::expect_ad(f, mu + rho * uv, mu, rho);
  }
}

TEST_F(AgradRev, mathMixScalFun_complex_uniform_lcdf_infinite_variate) {
  // a part of y that does not bind gives the derivative of any point beyond
  // the disk in that direction
  auto d_rho = [](const std::complex<double>& y) {
    stan::math::var rho = 1;
    stan::math::complex_uniform_lcdf(y, std::complex<double>(0, 0), rho).grad();
    return rho.adj();
  };
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FLOAT_EQ(d_rho({10, 0.3}), d_rho({inf, 0.3}));
  EXPECT_FLOAT_EQ(d_rho({0.3, 10}), d_rho({0.3, inf}));
}

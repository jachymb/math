#include <test/unit/math/test_ad.hpp>
#include <complex>
#include <vector>

TEST(mathMixFun, logicalEqComplex) {
  auto f = [](const auto& x1, const auto& x2) {
    return stan::math::logical_eq(x1, x2);
  };
  using c_t = std::complex<double>;
  std::vector<c_t> zs{c_t(1.5, -2), c_t(1.5, 2), c_t(2, 0), c_t(-0.5, 0)};
  for (const auto& z1 : zs) {
    for (const auto& z2 : zs) {
      stan::test::expect_complex_comparison(f, z1, z2);
    }
  }
}

TEST(mathMixFun, logicalEqComplexInt) {
  using stan::math::fvar;
  using stan::math::var;
  // stanc3 emits z == n and n == z for int n as logical_eq(z, n) and
  // logical_eq(n, z)
  auto expect_complex_int = [](const auto& zero) {
    using c_t = std::complex<std::decay_t<decltype(zero)>>;
    c_t z_eq(zero + 2, zero);
    c_t z_ne(zero + 2, zero + 1);
    EXPECT_TRUE(stan::math::logical_eq(z_eq, 2));
    EXPECT_TRUE(stan::math::logical_eq(2, z_eq));
    EXPECT_FALSE(stan::math::logical_eq(z_ne, 2));
    EXPECT_FALSE(stan::math::logical_eq(2, z_ne));
  };
  expect_complex_int(0.0);
  expect_complex_int(var(0));
  expect_complex_int(fvar<double>(0));
  expect_complex_int(fvar<fvar<double>>(0));
  expect_complex_int(fvar<var>(0));
  expect_complex_int(fvar<fvar<var>>(0));
}

#include <test/unit/math/test_ad.hpp>

TEST(mathMixCore, operatorMultiplication) {
  auto f = [](const auto& x1, const auto& x2) { return x1 * x2; };
  bool disable_lhs_int = true;
  stan::test::expect_common_binary(f, disable_lhs_int);
  stan::test::expect_complex_common_binary(f);
}

TEST(mathMixCore, operatorMultiplicationComplexInt) {
  // stanc3 emits z * n for complex z and int n as C++ infix under
  // `using namespace stan::math`; std has no complex<double> * int
  auto f = [](const auto& z) {
    using stan::math::operator*;
    return z * 3;
  };
  auto g = [](const auto& z) {
    using stan::math::operator*;
    return -2 * z;
  };
  std::complex<double> z(1.5, -0.5);
  stan::test::expect_ad(f, z);
  stan::test::expect_ad(g, z);
}

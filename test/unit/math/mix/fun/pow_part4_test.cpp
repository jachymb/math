#include <test/unit/math/test_ad.hpp>
#include <complex>
#include <vector>

TEST(mathMixFun, complexPowRowVectorMatrix) {
  using c_t = std::complex<double>;
  auto f
      = [](const auto& x1, const auto& x2) { return stan::math::pow(x1, x2); };
  stan::test::ad_tolerances tols;
  tols.hessian_hessian_ = 5e-3;
  tols.hessian_fvar_hessian_ = 5e-3;

  c_t z(1.2, -0.7);
  Eigen::RowVectorXcd rv1(2);
  rv1 << c_t(0.5, 1.2), c_t(1.4, -0.3);
  Eigen::RowVectorXcd rv2(2);
  rv2 << c_t(1.1, 0.8), c_t(-0.6, 1.1);
  Eigen::MatrixXcd m1(1, 2);
  m1 << c_t(0.5, 1.2), c_t(1.4, -0.3);
  Eigen::MatrixXcd m2(1, 2);
  m2 << c_t(1.1, 0.8), c_t(-0.6, 1.1);

  // x1 and x2 have the same type; covers (T, T), (T, complex), (complex, T)
  // and the same for array[] T
  auto expect_pow_shapes = [&](const auto& x1, const auto& x2) {
    using T = std::decay_t<decltype(x1)>;
    std::vector<T> a1{x1, x2};
    std::vector<T> a2{x2, x1};
    stan::test::expect_ad(tols, f, x1, x2);
    stan::test::expect_ad(tols, f, x1, z);
    stan::test::expect_ad(tols, f, z, x2);
    stan::test::expect_ad(tols, f, a1, a2);
    stan::test::expect_ad(tols, f, a1, z);
    stan::test::expect_ad(tols, f, z, a2);
  };
  expect_pow_shapes(rv1, rv2);
  expect_pow_shapes(m1, m2);

  // 1x2 matrices and an array[2, 1] keep the third-order finite
  // differences cheap; pow is elementwise, so size adds no coverage
  std::vector<std::vector<Eigen::MatrixXcd>> aam{{m1}, {m2}};
  stan::test::expect_ad(tols, f, aam, z);
}

TEST(mathMixFun, complexPowInt) {
  // the int exponent or base is bound in the lambda: expect_ad's int
  // overloads take only real arguments next to the int
  auto f = [](const auto& z) { return stan::math::pow(z, 3); };
  auto g = [](const auto& z) { return stan::math::pow(z, -2); };
  auto h = [](const auto& z) { return stan::math::pow(2, z); };
  stan::test::ad_tolerances tols;
  tols.hessian_hessian_ = 5e-3;
  tols.hessian_fvar_hessian_ = 5e-3;

  std::complex<double> z(1.2, -0.7);
  stan::test::expect_ad(tols, f, z);
  stan::test::expect_ad(tols, g, z);
  stan::test::expect_ad(tols, h, z);
}

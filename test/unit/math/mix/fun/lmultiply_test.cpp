#include <test/unit/math/test_ad.hpp>
#include <stan/math/rev/fun/lmultiply.hpp>
#include <limits>
#include <vector>

TEST(mathMixScalFun, lmultiply) {
  auto f = [](const auto& x1, const auto& x2) {
    return stan::math::lmultiply(x1, x2);
  };
  for (double x :
       std::vector<double>{-112.8, -15, 0, 0.5, 1.0, 2.0, 3.9, 157.2}) {
    for (double y : std::vector<double>{0.5, 1.0, 2.0, 3.9, 912.37}) {
      stan::test::expect_ad(f, x, y);
    }
  }

  double nan = std::numeric_limits<double>::quiet_NaN();
  stan::test::expect_ad(f, 1.0, nan);
  stan::test::expect_ad(f, nan, 1.0);
  stan::test::expect_ad(f, nan, nan);
}

// Second order at the same point: the derivative of x * log(x) is
// log(x) + 1, whose own derivative 1/x goes to +inf at 0. The reverse pass
// of the tangent also visits the value node, whose adjoint is 0 there, so
// its 0 * log(0) contribution has to be skipped for this to hold.
TEST(mathMixScalFun, lmultiply_second_derivative_at_zero) {
  using stan::math::fvar;
  using stan::math::var;
  constexpr double inf = std::numeric_limits<double>::infinity();
  fvar<var> x(0.0, 1.0);
  fvar<var> f = stan::math::lmultiply(x, x);
  EXPECT_EQ(0, f.val_.val());
  EXPECT_EQ(-inf, f.d_.val());
  f.d_.grad();
  EXPECT_EQ(inf, x.val_.adj());

  auto g = [](const auto& v) { return stan::math::lmultiply(v(0), v(0)); };
  Eigen::VectorXd v = Eigen::VectorXd::Zero(1);
  double fv;
  Eigen::VectorXd grad;
  Eigen::MatrixXd hess;
  stan::math::hessian(g, v, fv, grad, hess);
  EXPECT_EQ(0, fv);
  EXPECT_EQ(-inf, grad(0));
  EXPECT_EQ(inf, hess(0, 0));

  // the same through the (double, var) and (var, double) varis
  stan::math::recover_memory();
  fvar<var> y(0.0, 1.0);
  fvar<var> h = stan::math::lmultiply(2.0, y);  // 2 log y: 2 / y, -2 / y^2
  EXPECT_EQ(inf, h.d_.val());
  h.d_.grad();
  EXPECT_EQ(-inf, y.val_.adj());

  stan::math::recover_memory();
  fvar<var> y1(0.0, 1.0);
  fvar<var> h1 = stan::math::lmultiply(1.0, y1);  // log y: 1 / y, -1 / y^2
  EXPECT_EQ(inf, h1.d_.val());
  h1.d_.grad();
  EXPECT_EQ(-inf, y1.val_.adj());

  stan::math::recover_memory();
  fvar<var> z(1.0, 1.0);
  fvar<var> k = stan::math::lmultiply(z, 0.0);  // z log 0: log 0, 0
  EXPECT_EQ(-inf, k.d_.val());
  k.d_.grad();
  EXPECT_EQ(0, z.val_.adj());
}

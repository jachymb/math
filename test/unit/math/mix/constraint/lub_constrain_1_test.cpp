#include <test/unit/math/test_ad.hpp>
#include <test/unit/math/mix/constraint/lub_constrain_helpers.hpp>
#include <cmath>
#include <limits>
#include <vector>

TEST(mathMixMatFun, lub_constrain_scalars) {
  double x1 = 0.7;
  double x2 = -38.1;
  double lb = -2.0;
  double ub = 3.5;
  lub_constrain_tests::expect(x1, lb, ub);
  lub_constrain_tests::expect(x2, lb, ub);
  lub_constrain_tests::expect(x1, lb, lb);
  lub_constrain_tests::expect(x2, lb, lb);
  // ub inf
  auto ub_inf = stan::math::INFTY;
  lub_constrain_tests::expect(x1, lb, ub_inf);
  lub_constrain_tests::expect(x2, lb, ub_inf);

  // lb inf
  auto lb_inf = stan::math::NEGATIVE_INFTY;
  lub_constrain_tests::expect(x1, lb_inf, ub);
  lub_constrain_tests::expect(x2, lb_inf, ub);

  // both inf
  lub_constrain_tests::expect(x1, lb_inf, ub_inf);
  lub_constrain_tests::expect(x2, lb_inf, ub_inf);
}

namespace lub_constrain_fwd_test {
struct point {
  double x;
  double d1;
  double d2;
};
// mpmath, 60 digits: derivatives of -2 + 5 * inv_logit(x)
inline std::vector<point> points() {
  return {{-40, 2.1241771276457944e-17, 2.1241771276457944e-17},
          {-37, 4.2665238128720323e-16, 4.2665238128720313e-16},
          {-30, 4.678811484419212e-13, 4.678811484418336e-13},
          {-20, 1.0305768069709247e-08, 1.0305768027225704e-08},
          {-1, 0.9830596662074093, 0.45428873836474204},
          {0, 1.25, 0},
          {1, 0.9830596662074093, -0.45428873836474204},
          {20, 1.0305768069709247e-08, -1.0305768027225704e-08},
          {30, 4.678811484419212e-13, -4.678811484418336e-13},
          {37, 4.2665238128720323e-16, -4.2665238128720313e-16},
          {40, 2.1241771276457944e-17, -2.1241771276457944e-17}};
}
inline void expect_rel(double expected, double actual, double x) {
  EXPECT_NEAR(
      expected, actual,
      1e-13
          * std::fmax(std::fabs(expected), std::numeric_limits<double>::min()))
      << "x = " << x;
}
}  // namespace lub_constrain_fwd_test

TEST(mathMixScalFun, lub_constrain_fwd_derivative_tails) {
  using lub_constrain_fwd_test::expect_rel;
  using stan::math::fvar;
  using stan::math::var;
  for (auto p : lub_constrain_fwd_test::points()) {
    expect_rel(p.d1,
               stan::math::lub_constrain(fvar<double>(p.x, 1), -2.0, 3.0).d_,
               p.x);

    fvar<fvar<double>> ff(fvar<double>(p.x, 1), fvar<double>(1, 0));
    fvar<fvar<double>> y_ff = stan::math::lub_constrain(ff, -2.0, 3.0);
    expect_rel(p.d1, y_ff.d_.val_, p.x);
    expect_rel(p.d2, y_ff.d_.d_, p.x);

    fvar<var> fv(p.x, 1);
    fvar<var> y_fv = stan::math::lub_constrain(fv, -2.0, 3.0);
    expect_rel(p.d1, y_fv.d_.val(), p.x);
    y_fv.d_.grad();
    expect_rel(p.d2, fv.val_.adj(), p.x);
    stan::math::recover_memory();
  }
}

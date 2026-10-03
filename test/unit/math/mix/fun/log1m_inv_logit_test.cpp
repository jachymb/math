#include <test/unit/math/test_ad.hpp>
#include <cmath>
#include <limits>
#include <vector>

TEST(mathMixMatFun, log1mInvLogit) {
  auto f = [](const auto& x1) { return stan::math::log1m_inv_logit(x1); };
  stan::test::expect_common_nonzero_unary_vectorized<
      stan::test::ScalarSupport::Real>(f);
  stan::test::expect_unary_vectorized(f, -2.6, -2, -1, -0.5, -0.2, 0.5, 1, 1.3,
                                      3, 5);
}

TEST(mathMixMatFun, log1minvlogit_varmat) {
  using stan::math::vec_concat;
  using stan::test::expect_ad_vector_matvar;
  using stan::test::internal::common_args;
  auto f = [](const auto& x1) {
    using stan::math::log1m_inv_logit;
    return log1m_inv_logit(x1);
  };
  std::vector<double> com_args = common_args();
  std::vector<double> args{-2.6, -2, -1, -0.5, -0.2, 0.5, 1, 1.3, 3, 5};
  auto all_args = vec_concat(com_args, args);
  Eigen::VectorXd A(all_args.size());
  for (int i = 0; i < all_args.size(); ++i) {
    A(i) = all_args[i];
  }
  expect_ad_vector_matvar(f, A);
}

namespace log1m_inv_logit_test {
struct point {
  double x;
  double d1;
  double d2;
};
// mpmath, 60 digits: d1 = -inv_logit(x), d2 = -inv_logit'(x)
inline std::vector<point> points() {
  return {{-40, -4.248354255291589e-18, -4.248354255291589e-18},
          {-37, -8.533047625744065e-17, -8.533047625744065e-17},
          {-30, -9.357622968839299e-14, -9.357622968838423e-14},
          {-20, -2.0611536181902037e-09, -2.061153613941849e-09},
          {-1, -0.2689414213699951, -0.19661193324148185},
          {0, -0.5, -0.25},
          {1, -0.7310585786300049, -0.19661193324148185},
          {20, -0.9999999979388464, -2.061153613941849e-09},
          {30, -0.9999999999999064, -9.357622968838423e-14},
          {37, -0.9999999999999999, -8.533047625744065e-17},
          {40, -1.0, -4.248354255291589e-18}};
}
inline void expect_rel(double expected, double actual, double x) {
  EXPECT_NEAR(
      expected, actual,
      1e-13
          * std::fmax(std::fabs(expected), std::numeric_limits<double>::min()))
      << "x = " << x;
}
}  // namespace log1m_inv_logit_test

TEST(mathMixScalFun, log1mInvLogitFwdDerivativeTails) {
  using log1m_inv_logit_test::expect_rel;
  using stan::math::fvar;
  using stan::math::var;
  for (auto p : log1m_inv_logit_test::points()) {
    expect_rel(p.d1, stan::math::log1m_inv_logit(fvar<double>(p.x, 1)).d_, p.x);

    fvar<fvar<double>> ff(fvar<double>(p.x, 1), fvar<double>(1, 0));
    fvar<fvar<double>> y_ff = stan::math::log1m_inv_logit(ff);
    expect_rel(p.d1, y_ff.d_.val_, p.x);
    expect_rel(p.d2, y_ff.d_.d_, p.x);

    fvar<var> fv(p.x, 1);
    fvar<var> y_fv = stan::math::log1m_inv_logit(fv);
    expect_rel(p.d1, y_fv.d_.val(), p.x);
    y_fv.d_.grad();
    expect_rel(p.d2, fv.val_.adj(), p.x);
    stan::math::recover_memory();
  }
}

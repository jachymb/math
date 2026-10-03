#include <test/unit/math/test_ad.hpp>
#include <cmath>
#include <limits>
#include <vector>

TEST(mathMixMatFun, invLogit) {
  auto f = [](const auto& x1) { return stan::math::inv_logit(x1); };
  stan::test::expect_common_unary_vectorized(f);
  stan::test::expect_unary_vectorized(f, -2.6, -2, -1.2, -0.2, 0.5, 1, 1.3, 1.5,
                                      3);

  std::vector<double> com_args = stan::test::internal::common_nonzero_args();
  std::vector<double> args{-2.6, -0.5, 0.5, 1.5};

  stan::test::expect_ad_vector_matvar(f, stan::math::to_vector(com_args));
  stan::test::expect_ad_vector_matvar(f, stan::math::to_vector(args));
}

namespace inv_logit_test {
// closed forms with e = exp(-|x|): no 1 - inv_logit(x) cancellation
inline double d1(double x) {
  double e = std::exp(-std::fabs(x));
  return e / ((1 + e) * (1 + e));
}
inline double d2(double x) {
  double e = std::exp(-std::fabs(x));
  return (x > 0 ? 1 : -1) * d1(x) * std::expm1(-std::fabs(x)) / (1 + e);
}
inline void expect_rel(double actual, double expected, double x) {
  EXPECT_NEAR(
      actual, expected,
      1e-13
          * std::fmax(std::fabs(expected), std::numeric_limits<double>::min()))
      << "x = " << x;
}
inline std::vector<double> args() {
  return {-800, -40, -20, -1, 0, 1, 20, 30, 37, 40, 100, 700, 800};
}
}  // namespace inv_logit_test

TEST(mathMixScalFun, invLogitDerivativesLargeArgs) {
  using inv_logit_test::d1;
  using inv_logit_test::d2;
  using inv_logit_test::expect_rel;
  using stan::math::fvar;
  using stan::math::inv_logit;
  using stan::math::var;
  for (double x : inv_logit_test::args()) {
    var a = x;
    inv_logit(a).grad();
    expect_rel(a.adj(), d1(x), x);
    stan::math::recover_memory();

    expect_rel(inv_logit(fvar<double>(x, 1)).d_, d1(x), x);

    fvar<fvar<double>> ff(fvar<double>(x, 1), fvar<double>(1, 0));
    expect_rel(inv_logit(ff).d_.d_, d2(x), x);

    fvar<var> fv(x, 1);
    inv_logit(fv).d_.grad();
    expect_rel(fv.val_.adj(), d2(x), x);
    stan::math::recover_memory();
  }
}

TEST(mathMixMatFun, invLogitDerivativesLargeArgs) {
  using inv_logit_test::d1;
  using inv_logit_test::expect_rel;
  using stan::math::inv_logit;
  using stan::math::sum;
  using stan::math::var;
  std::vector<double> xs = inv_logit_test::args();
  Eigen::VectorXd x = stan::math::to_vector(xs);

  stan::math::var_value<Eigen::VectorXd> xv(x);
  sum(inv_logit(xv)).grad();
  for (int i = 0; i < x.size(); ++i) {
    expect_rel(xv.adj()(i), d1(x(i)), x(i));
  }
  stan::math::recover_memory();

  Eigen::Matrix<var, Eigen::Dynamic, 1> xm = x;
  sum(inv_logit(xm)).grad();
  for (int i = 0; i < x.size(); ++i) {
    expect_rel(xm(i).adj(), d1(x(i)), x(i));
  }
  stan::math::recover_memory();

  std::vector<var> xs_v(xs.begin(), xs.end());
  sum(inv_logit(xs_v)).grad();
  for (size_t i = 0; i < xs.size(); ++i) {
    expect_rel(xs_v[i].adj(), d1(xs[i]), xs[i]);
  }
  stan::math::recover_memory();
}

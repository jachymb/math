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
struct point {
  double x;
  double d1;  // inv_logit'(x)
  double d2;  // inv_logit''(x)
};
// mpmath, 50 digits; 0 where the value underflows (e^-800)
inline std::vector<point> points() {
  return {{-800, 0, 0},
          {-40, 4.2483542552915889592e-18, 4.2483542552915889231e-18},
          {-20, 2.0611536139418493437e-9, 2.0611536054451408856e-9},
          {-1, 1.9661193324148185254e-1, 9.0857747672948409442e-2},
          {0, 0.25, 0},
          {1, 1.9661193324148185254e-1, -9.0857747672948409442e-2},
          {20, 2.0611536139418493437e-9, -2.0611536054451408856e-9},
          {30, 9.3576229688384233028e-14, -9.3576229688366720006e-14},
          {37, 8.533047625744064338e-17, -8.5330476257440628818e-17},
          {40, 4.2483542552915889592e-18, -4.2483542552915889231e-18},
          {100, 3.720075976020835963e-44, -3.720075976020835963e-44},
          {700, 9.8596765437597708567e-305, -9.8596765437597708567e-305},
          {800, 0, 0}};
}
inline void expect_rel(double actual, double expected, double x) {
  EXPECT_NEAR(
      actual, expected,
      1e-13
          * std::fmax(std::fabs(expected), std::numeric_limits<double>::min()))
      << "x = " << x;
}
}  // namespace inv_logit_test

TEST(mathMixScalFun, invLogitDerivativesLargeArgs) {
  using inv_logit_test::expect_rel;
  using stan::math::fvar;
  using stan::math::inv_logit;
  using stan::math::var;
  for (auto p : inv_logit_test::points()) {
    var a = p.x;
    inv_logit(a).grad();
    expect_rel(a.adj(), p.d1, p.x);
    stan::math::recover_memory();

    expect_rel(inv_logit(fvar<double>(p.x, 1)).d_, p.d1, p.x);

    fvar<fvar<double>> ff(fvar<double>(p.x, 1), fvar<double>(1, 0));
    expect_rel(inv_logit(ff).d_.d_, p.d2, p.x);

    fvar<var> fv(p.x, 1);
    inv_logit(fv).d_.grad();
    expect_rel(fv.val_.adj(), p.d2, p.x);
    stan::math::recover_memory();
  }
}

TEST(mathMixMatFun, invLogitDerivativesLargeArgs) {
  using inv_logit_test::expect_rel;
  using stan::math::fvar;
  using stan::math::inv_logit;
  using stan::math::sum;
  using stan::math::var;
  std::vector<inv_logit_test::point> ps = inv_logit_test::points();
  int n = ps.size();
  Eigen::VectorXd x(n);
  Eigen::VectorXd d1(n);
  Eigen::VectorXd d2(n);
  for (int i = 0; i < n; ++i) {
    x(i) = ps[i].x;
    d1(i) = ps[i].d1;
    d2(i) = ps[i].d2;
  }

  stan::math::var_value<Eigen::VectorXd> xv(x);
  sum(inv_logit(xv)).grad();
  for (int i = 0; i < n; ++i) {
    expect_rel(xv.adj()(i), d1(i), x(i));
  }
  stan::math::recover_memory();

  stan::math::var_value<Eigen::RowVectorXd> xr(x.transpose());
  sum(inv_logit(xr)).grad();
  for (int i = 0; i < n; ++i) {
    expect_rel(xr.adj()(i), d1(i), x(i));
  }
  stan::math::recover_memory();

  // inv_logit' is even
  Eigen::MatrixXd x2(n, 2);
  x2 << x, -x;
  stan::math::var_value<Eigen::MatrixXd> xmv(x2);
  sum(inv_logit(xmv)).grad();
  for (int i = 0; i < n; ++i) {
    expect_rel(xmv.adj()(i, 0), d1(i), x(i));
    expect_rel(xmv.adj()(i, 1), d1(i), -x(i));
  }
  stan::math::recover_memory();

  Eigen::Matrix<var, Eigen::Dynamic, 1> xm = x;
  sum(inv_logit(xm)).grad();
  for (int i = 0; i < n; ++i) {
    expect_rel(xm(i).adj(), d1(i), x(i));
  }
  stan::math::recover_memory();

  std::vector<var> xs(x.data(), x.data() + n);
  sum(inv_logit(xs)).grad();
  for (int i = 0; i < n; ++i) {
    expect_rel(xs[i].adj(), d1(i), x(i));
  }
  stan::math::recover_memory();

  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> xf(n);
  for (int i = 0; i < n; ++i) {
    xf(i) = fvar<var>(x(i), 1);
  }
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> yf = inv_logit(xf);
  for (int i = 0; i < n; ++i) {
    expect_rel(yf(i).d_.val(), d1(i), x(i));
  }
  sum(yf).d_.grad();
  for (int i = 0; i < n; ++i) {
    expect_rel(xf(i).val_.adj(), d2(i), x(i));
  }
  stan::math::recover_memory();
}

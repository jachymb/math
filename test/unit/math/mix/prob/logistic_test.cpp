#include <stan/math/mix.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/math/mix/prob/logistic_tail_refs.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace logistic_test {
using logistic_tail_refs::logistic_ref;
using logistic_tail_refs::mu;
using logistic_tail_refs::sigma;

inline void expect_rel(double expected, double actual, const std::string& msg,
                       double tol = 1e-12) {
  EXPECT_NEAR(
      expected, actual,
      tol * std::fmax(std::fabs(expected), std::numeric_limits<double>::min()))
      << msg;
}

auto lpdf = [](const auto& y, const auto& m, const auto& s) {
  return stan::math::logistic_lpdf(y, m, s);
};
auto cdf = [](const auto& y, const auto& m, const auto& s) {
  return stan::math::logistic_cdf(y, m, s);
};
auto lcdf = [](const auto& y, const auto& m, const auto& s) {
  return stan::math::logistic_lcdf(y, m, s);
};
auto lccdf = [](const auto& y, const auto& m, const auto& s) {
  return stan::math::logistic_lccdf(y, m, s);
};

template <typename F, size_t N>
void expect_var_refs(const F& f, const logistic_ref (&refs)[N]) {
  using stan::math::var;
  for (const auto& r : refs) {
    const std::string msg = "z = " + std::to_string(r.z);
    var y = mu + sigma * r.z;
    var m = mu;
    var s = sigma;
    var lp = f(y, m, s);
    lp.grad();
    expect_rel(r.val, lp.val(), msg);
    expect_rel(r.dy, y.adj(), msg);
    expect_rel(r.dmu, m.adj(), msg);
    expect_rel(r.dsigma, s.adj(), msg);
    stan::math::recover_memory();
  }
}

template <typename F, size_t N>
void expect_fvar_refs(const F& f, const logistic_ref (&refs)[N]) {
  using stan::math::fvar;
  for (const auto& r : refs) {
    const std::string msg = "z = " + std::to_string(r.z);
    const double y = mu + sigma * r.z;
    fvar<double> lp_y = f(fvar<double>(y, 1), fvar<double>(mu), sigma);
    fvar<double> lp_mu = f(y, fvar<double>(mu, 1), fvar<double>(sigma));
    fvar<double> lp_sigma = f(fvar<double>(y), mu, fvar<double>(sigma, 1));
    expect_rel(r.val, lp_y.val_, msg);
    expect_rel(r.dy, lp_y.d_, msg);
    expect_rel(r.dmu, lp_mu.d_, msg);
    expect_rel(r.dsigma, lp_sigma.d_, msg);
  }
}

// Second derivative in y through fvar<fvar<double>> and fvar<var>.
template <typename F, size_t N>
void expect_second_refs(const F& f, const logistic_ref (&refs)[N]) {
  using logistic_tail_refs::d2y;
  using stan::math::fvar;
  using stan::math::var;
  static_assert(N == sizeof(d2y) / sizeof(d2y[0]), "grid mismatch");
  for (size_t i = 0; i < N; ++i) {
    const std::string msg = "z = " + std::to_string(refs[i].z);
    const double y = mu + sigma * refs[i].z;
    fvar<fvar<double>> y_ff(fvar<double>(y, 1), fvar<double>(1, 0));
    fvar<fvar<double>> lp_ff = f(y_ff, mu, sigma);
    expect_rel(refs[i].dy, lp_ff.d_.val_, msg, 1e-13);
    expect_rel(d2y[i], lp_ff.d_.d_, msg, 1e-13);

    fvar<var> y_fv(y, 1);
    fvar<var> lp_fv = f(y_fv, mu, sigma);
    expect_rel(refs[i].dy, lp_fv.d_.val(), msg, 1e-13);
    lp_fv.d_.grad();
    expect_rel(d2y[i], y_fv.val_.adj(), msg, 1e-13);
    stan::math::recover_memory();
  }
}

// All arguments vectors, then y a vector with mu and sigma broadcast.
template <typename F, size_t N>
void expect_vector_refs(const F& f, const logistic_ref (&refs)[N]) {
  using stan::math::var;
  std::vector<var> y;
  std::vector<var> m;
  std::vector<var> s;
  double val = 0;
  double val_abs = 0;
  double dmu = 0;
  double dmu_abs = 0;
  double dsigma = 0;
  double dsigma_abs = 0;
  for (const auto& r : refs) {
    y.push_back(mu + sigma * r.z);
    m.push_back(mu);
    s.push_back(sigma);
    val += r.val;
    val_abs += std::fabs(r.val);
    dmu += r.dmu;
    dmu_abs += std::fabs(r.dmu);
    dsigma += r.dsigma;
    dsigma_abs += std::fabs(r.dsigma);
  }
  var lp = f(y, m, s);
  lp.grad();
  EXPECT_NEAR(val, lp.val(), 1e-12 * val_abs);
  for (size_t i = 0; i < N; ++i) {
    const std::string msg = "z = " + std::to_string(refs[i].z);
    expect_rel(refs[i].dy, y[i].adj(), msg);
    expect_rel(refs[i].dmu, m[i].adj(), msg);
    expect_rel(refs[i].dsigma, s[i].adj(), msg);
  }
  stan::math::set_zero_all_adjoints();

  var m_scal = mu;
  var s_scal = sigma;
  lp = f(y, m_scal, s_scal);
  lp.grad();
  EXPECT_NEAR(val, lp.val(), 1e-12 * val_abs);
  for (size_t i = 0; i < N; ++i) {
    expect_rel(refs[i].dy, y[i].adj(), "z = " + std::to_string(refs[i].z));
  }
  EXPECT_NEAR(dmu, m_scal.adj(), 1e-12 * dmu_abs);
  EXPECT_NEAR(dsigma, s_scal.adj(), 1e-12 * dsigma_abs);
  stan::math::recover_memory();

  // y a scalar with mu and sigma vectors: mu_i = y - sigma * z_i.
  double dy = 0;
  double dy_abs = 0;
  for (const auto& r : refs) {
    dy += r.dy;
    dy_abs += std::fabs(r.dy);
  }
  var y_scal = mu;
  std::vector<var> m_vec;
  std::vector<var> s_vec;
  for (const auto& r : refs) {
    m_vec.push_back(mu - sigma * r.z);
    s_vec.push_back(sigma);
  }
  lp = f(y_scal, m_vec, s_vec);
  lp.grad();
  EXPECT_NEAR(val, lp.val(), 1e-12 * val_abs);
  EXPECT_NEAR(dy, y_scal.adj(), 1e-12 * dy_abs);
  for (size_t i = 0; i < N; ++i) {
    const std::string msg = "z = " + std::to_string(refs[i].z);
    expect_rel(refs[i].dmu, m_vec[i].adj(), msg);
    expect_rel(refs[i].dsigma, s_vec[i].adj(), msg);
  }
  stan::math::recover_memory();

  // fvar: y a vector with unit tangents.
  std::vector<stan::math::fvar<double>> y_fv;
  for (const auto& r : refs) {
    y_fv.emplace_back(mu + sigma * r.z, 1.0);
  }
  stan::math::fvar<double> lp_fv = f(y_fv, mu, sigma);
  EXPECT_NEAR(val, lp_fv.val_, 1e-12 * val_abs);
  EXPECT_NEAR(dy, lp_fv.d_, 1e-12 * dy_abs);
}
}  // namespace logistic_test

TEST_F(AgradRev, mathMixScalFun_logistic_lpdf_tails) {
  logistic_test::expect_var_refs(logistic_test::lpdf, logistic_tail_refs::lpdf);
  logistic_test::expect_fvar_refs(logistic_test::lpdf,
                                  logistic_tail_refs::lpdf);
  logistic_test::expect_vector_refs(logistic_test::lpdf,
                                    logistic_tail_refs::lpdf);
}

TEST_F(AgradRev, mathMixScalFun_logistic_lpdf_large_location) {
  using stan::math::var;
  // mpmath at 60 digits; z = 1 with |mu / sigma| = 800
  for (double m_dbl : {800.0, -800.0}) {
    var y = m_dbl + 1;
    var m = m_dbl;
    var s = 1;
    var lp = stan::math::logistic_lpdf(y, m, s);
    lp.grad();
    logistic_test::expect_rel(-1.6265233750364457, lp.val(), "value");
    logistic_test::expect_rel(-0.46211715726000976, y.adj(), "y");
    logistic_test::expect_rel(0.46211715726000976, m.adj(), "mu");
    logistic_test::expect_rel(-0.53788284273999024, s.adj(), "sigma");
    stan::math::recover_memory();
  }
}

TEST_F(AgradRev, mathMixScalFun_logistic_lpdf_hessian_tails) {
  using logistic_tail_refs::mu;
  using logistic_tail_refs::sigma;
  using stan::math::fvar;
  using stan::math::var;
  // Hessian in (y, mu, sigma), mpmath at 80 digits; the y and mu block is
  // -sech(z / 2)^2 / (2 sigma^2), which underflows to 0.
  for (double z : {-800.0, 800.0}) {
    const double t = z > 0 ? 0.25 : -0.25;
    const double hess[3][3] = {{0, 0, t}, {0, 0, -t}, {t, -t, -399.75}};
    for (int j = 0; j < 3; ++j) {
      fvar<var> y(mu + sigma * z, j == 0);
      fvar<var> m(mu, j == 1);
      fvar<var> s(sigma, j == 2);
      fvar<var> lp = stan::math::logistic_lpdf(y, m, s);
      lp.d_.grad();
      const std::string msg
          = "z = " + std::to_string(z) + ", row " + std::to_string(j);
      EXPECT_EQ(hess[j][0], y.val_.adj()) << msg;
      EXPECT_EQ(hess[j][1], m.val_.adj()) << msg;
      logistic_test::expect_rel(hess[j][2], s.val_.adj(), msg);
      stan::math::recover_memory();
    }
    // mu the only autodiff argument
    fvar<var> m(mu, 1);
    fvar<var> lp = stan::math::logistic_lpdf(mu + sigma * z, m, sigma);
    lp.d_.grad();
    EXPECT_EQ(0.0, m.val_.adj()) << "z = " << z << ", mu only";
    stan::math::recover_memory();
  }
}

TEST_F(AgradRev, mathMixScalFun_logistic_lpdf_near_zero) {
  using stan::math::var;
  // The y and mu partials are -+tanh(z / 2) / sigma; here z = 1e-10.
  var y = 2e-10;
  var m = 0;
  var s = 2;
  var lp = stan::math::logistic_lpdf(y, m, s);
  lp.grad();
  logistic_test::expect_rel(-2.5e-11, y.adj(), "y", 1e-14);
  logistic_test::expect_rel(2.5e-11, m.adj(), "mu", 1e-14);
}

TEST_F(AgradRev, mathMixScalFun_logistic_cdf_tails) {
  using logistic_tail_refs::lcdf;
  using logistic_tail_refs::mu;
  using logistic_tail_refs::sigma;
  using stan::math::var;
  logistic_test::expect_var_refs(logistic_test::cdf, logistic_tail_refs::cdf);
  logistic_test::expect_fvar_refs(logistic_test::cdf, logistic_tail_refs::cdf);

  // The product of the cdfs underflows to 0 with z = -800 included: the value
  // and all partials are 0.
  std::vector<var> y;
  for (const auto& r : lcdf) {
    y.push_back(mu + sigma * r.z);
  }
  var m = mu;
  var s = sigma;
  var p = stan::math::logistic_cdf(y, m, s);
  p.grad();
  EXPECT_EQ(0.0, p.val());
  for (const auto& y_i : y) {
    EXPECT_EQ(0.0, y_i.adj());
  }
  EXPECT_EQ(0.0, m.adj());
  EXPECT_EQ(0.0, s.adj());
  stan::math::recover_memory();

  // Without it, d P / d y_i = P * d lcdf / d y_i.
  y.clear();
  double p_ref = 1;
  for (size_t i = 1; i < 12; ++i) {
    y.push_back(mu + sigma * lcdf[i].z);
    p_ref *= logistic_tail_refs::cdf[i].val;
  }
  var p2 = stan::math::logistic_cdf(y, m, s);
  p2.grad();
  logistic_test::expect_rel(p_ref, p2.val(), "value");
  for (size_t i = 1; i < 12; ++i) {
    logistic_test::expect_rel(p_ref * lcdf[i].dy, y[i - 1].adj(),
                              "z = " + std::to_string(lcdf[i].z));
  }
}

TEST_F(AgradRev, mathMixScalFun_logistic_lcdf_tails) {
  logistic_test::expect_var_refs(logistic_test::lcdf, logistic_tail_refs::lcdf);
  logistic_test::expect_fvar_refs(logistic_test::lcdf,
                                  logistic_tail_refs::lcdf);
  logistic_test::expect_vector_refs(logistic_test::lcdf,
                                    logistic_tail_refs::lcdf);
}

TEST_F(AgradRev, mathMixScalFun_logistic_lcdf_second_derivative_tails) {
  logistic_test::expect_second_refs(logistic_test::lcdf,
                                    logistic_tail_refs::lcdf);
}

TEST_F(AgradRev, mathMixScalFun_logistic_lccdf_second_derivative_tails) {
  logistic_test::expect_second_refs(logistic_test::lccdf,
                                    logistic_tail_refs::lccdf);
}

TEST_F(AgradRev, mathMixScalFun_logistic_lccdf_tails) {
  logistic_test::expect_var_refs(logistic_test::lccdf,
                                 logistic_tail_refs::lccdf);
  logistic_test::expect_fvar_refs(logistic_test::lccdf,
                                  logistic_tail_refs::lccdf);
  logistic_test::expect_vector_refs(logistic_test::lccdf,
                                    logistic_tail_refs::lccdf);
}

TEST_F(AgradRev, mathMixScalFun_logistic_infinite_y) {
  using stan::math::var;
  const double inf = std::numeric_limits<double>::infinity();
  auto expect = [](const auto& f, double y_dbl, double expected) {
    var y = y_dbl;
    var m = 0.5;
    var s = 2;
    var lp = f(y, m, s);
    lp.grad();
    EXPECT_EQ(expected, lp.val());
    EXPECT_EQ(0.0, y.adj());
    EXPECT_EQ(0.0, m.adj());
    EXPECT_EQ(0.0, s.adj());
    stan::math::set_zero_all_adjoints();
  };
  expect(logistic_test::cdf, -inf, 0);
  expect(logistic_test::cdf, inf, 1);
  expect(logistic_test::lcdf, -inf, -inf);
  expect(logistic_test::lcdf, inf, 0);
  expect(logistic_test::lccdf, -inf, 0);
  expect(logistic_test::lccdf, inf, -inf);

  // Behaviour change: develop returned -inf here with the partials of the
  // elements before y = inf left non-zero.
  std::vector<var> y{1.0, inf};
  var m = 0.5;
  var s = 2;
  var lp = stan::math::logistic_lccdf(y, m, s);
  lp.grad();
  EXPECT_EQ(-inf, lp.val());
  EXPECT_EQ(0.0, y[0].adj());
  EXPECT_EQ(0.0, m.adj());
  EXPECT_EQ(0.0, s.adj());

  // +0, not -0
  EXPECT_FALSE(std::signbit(stan::math::logistic_lcdf(inf, 0.5, 2.0)));
  EXPECT_FALSE(std::signbit(
      stan::math::logistic_lcdf(std::vector<double>{inf}, 0.5, 2.0)));
  EXPECT_FALSE(std::signbit(
      stan::math::logistic_lccdf(std::vector<double>{-1e4}, 0.5, 2.0)));
}

TEST_F(AgradRev, mathMixScalFun_logistic_infinite_z) {
  using stan::math::var;
  const double inf = std::numeric_limits<double>::infinity();
  // Finite y and mu with (y - mu) / sigma overflowing to -inf, then +inf;
  // the sigma partial is inf * 0 there, set to 0.
  auto expect = [](const auto& f, double y_dbl, double expected) {
    var y = y_dbl;
    var m = -y_dbl;
    var s = 1;
    var lp = f(y, m, s);
    lp.grad();
    EXPECT_EQ(expected, lp.val());
    EXPECT_FALSE(std::isnan(y.adj()));
    EXPECT_FALSE(std::isnan(m.adj()));
    EXPECT_EQ(0.0, s.adj());
    stan::math::set_zero_all_adjoints();
  };
  expect(logistic_test::cdf, -1e308, 0);
  expect(logistic_test::cdf, 1e308, 1);
  expect(logistic_test::lcdf, -1e308, -inf);
  expect(logistic_test::lcdf, 1e308, 0);
  expect(logistic_test::lccdf, -1e308, 0);
  expect(logistic_test::lccdf, 1e308, -inf);
}

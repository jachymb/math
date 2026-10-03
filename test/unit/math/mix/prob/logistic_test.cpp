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
  EXPECT_NEAR(expected, actual, tol * std::fabs(expected)) << msg;
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
}

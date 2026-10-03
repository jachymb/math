#ifndef TEST_UNIT_MATH_OPENCL_REV_LOGISTIC_TAIL_REFS_CL_HPP
#define TEST_UNIT_MATH_OPENCL_REV_LOGISTIC_TAIL_REFS_CL_HPP
#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <test/unit/math/mix/prob/logistic_tail_refs.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace logistic_tail_refs {

// Value and partials of f on the device at y = loc + scale * z for each
// reference, with one-element var_value<matrix_cl<double>> arguments.
template <typename F, size_t N>
void expect_opencl_refs(const F& f, const logistic_ref (&refs)[N],
                        double loc = mu, double scale = sigma) {
  using stan::math::from_matrix_cl;
  using stan::math::matrix_cl;
  using stan::math::var;
  using stan::math::var_value;
  auto expect_rel = [](double expected, double actual, double z) {
    EXPECT_NEAR(expected, actual, 1e-12 * std::fabs(expected)) << "z = " << z;
  };
  for (const auto& r : refs) {
    Eigen::VectorXd y_val = Eigen::VectorXd::Constant(1, loc + scale * r.z);
    Eigen::VectorXd mu_val = Eigen::VectorXd::Constant(1, loc);
    Eigen::VectorXd sigma_val = Eigen::VectorXd::Constant(1, scale);
    var_value<matrix_cl<double>> y = matrix_cl<double>(y_val);
    var_value<matrix_cl<double>> m = matrix_cl<double>(mu_val);
    var_value<matrix_cl<double>> s = matrix_cl<double>(sigma_val);
    var lp = f(y, m, s);
    lp.grad();
    expect_rel(r.val, lp.val(), r.z);
    expect_rel(r.dy, from_matrix_cl(y.adj())(0), r.z);
    expect_rel(r.dmu, from_matrix_cl(m.adj())(0), r.z);
    expect_rel(r.dsigma, from_matrix_cl(s.adj())(0), r.z);
    stan::math::recover_memory();
  }
}

// y and mu finite with (y - mu) / sigma overflowing to -inf and +inf: no
// NaN partials and a sigma partial of 0.
template <typename F>
void expect_opencl_inf_z(const F& f) {
  using stan::math::from_matrix_cl;
  using stan::math::matrix_cl;
  using stan::math::var;
  using stan::math::var_value;
  for (double y_dbl : {-1e308, 1e308}) {
    Eigen::VectorXd y_val = Eigen::VectorXd::Constant(1, y_dbl);
    Eigen::VectorXd mu_val = -y_val;
    Eigen::VectorXd sigma_val = Eigen::VectorXd::Constant(1, 1.0);
    var_value<matrix_cl<double>> y = matrix_cl<double>(y_val);
    var_value<matrix_cl<double>> m = matrix_cl<double>(mu_val);
    var_value<matrix_cl<double>> s = matrix_cl<double>(sigma_val);
    var lp = f(y, m, s);
    lp.grad();
    EXPECT_FALSE(std::isnan(from_matrix_cl(y.adj())(0))) << "y = " << y_dbl;
    EXPECT_FALSE(std::isnan(from_matrix_cl(m.adj())(0))) << "y = " << y_dbl;
    EXPECT_EQ(0.0, from_matrix_cl(s.adj())(0)) << "y = " << y_dbl;
    stan::math::recover_memory();
  }
}

}  // namespace logistic_tail_refs

#endif
#endif

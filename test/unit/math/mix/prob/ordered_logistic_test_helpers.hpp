#ifndef TEST_UNIT_MATH_MIX_PROB_ORDERED_LOGISTIC_TEST_HELPERS_HPP
#define TEST_UNIT_MATH_MIX_PROB_ORDERED_LOGISTIC_TEST_HELPERS_HPP

#include <stan/math/mix.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

/**
 * Derivative checks for ordered_logistic_glm_lpmf and ordered_logistic_lpmf
 * against closed forms, up to third order and for every autodiff nesting.
 *
 * The parameters are v = (beta, cuts). An observation of class k with
 * location theta = x_n beta has, with a = theta - c_{k-1} and
 * b = theta - c_k (c_0 = -inf, c_K = +inf),
 *
 *   log(inv_logit(a) - inv_logit(b))
 *     = log inv_logit(a) + log inv_logit(-b) + log(1 - exp(-(c_k - c_{k-1}))),
 *
 * a sum of univariate functions of linear forms in v (the first term is
 * absent for k = 1, the second for k = K, the third for both). For
 * p = inv_logit(t), q = 1 - p, the first two have derivatives
 * (q, -p q, -p q (q - p)) at t = a and (-p, -p q, -p q (q - p)) at t = b, and
 * with r = 1 / expm1(c_k - c_{k-1}) the third has (r, -r (1 + r),
 * r (1 + r) (1 + 2 r)). Each tolerance is relative to the same sum taken over
 * absolute values.
 */
namespace ordered_logistic_test {

/** One univariate summand f(l . v): value and derivatives 1 to 3. */
struct term {
  Eigen::VectorXd l;
  double f[4];
};

template <typename T_y, typename T_x>
std::vector<term> terms(const T_y& y, const T_x& x, const Eigen::VectorXd& beta,
                        const Eigen::VectorXd& cuts) {
  using stan::math::inv_logit;
  const int K = beta.size();
  const int C = cuts.size();
  const int N = std::max<int>(x.rows(), stan::math::size(y));
  std::vector<term> out;
  for (int n = 0; n < N; ++n) {
    const int k = stan::math::size(y) == 1 ? stan::get(y, 0) : stan::get(y, n);
    Eigen::VectorXd xn = x.row(x.rows() == 1 ? 0 : n).transpose();
    const double theta = xn.dot(beta);
    for (int side = 0; side < 2; ++side) {
      const int i = side == 0 ? k - 2 : k - 1;  // index of the cut
      if (i < 0 || i >= C) {
        continue;
      }
      const double t = theta - cuts(i);
      const double p = inv_logit(t), q = inv_logit(-t);
      term tm{Eigen::VectorXd::Zero(K + C), {0, 0, -p * q, -p * q * (q - p)}};
      tm.l.head(K) = xn;
      tm.l(K + i) = -1;
      tm.f[0] = side == 0 ? stan::math::log_inv_logit(t)
                          : stan::math::log_inv_logit(-t);
      tm.f[1] = side == 0 ? q : -p;
      out.push_back(tm);
    }
    if (k > 1 && k <= C) {
      const double d = cuts(k - 1) - cuts(k - 2);
      const double r = 1 / std::expm1(d);
      term tm{Eigen::VectorXd::Zero(K + C),
              {stan::math::log1m_exp(-d), r, -r * (1 + r),
               r * (1 + r) * (1 + 2 * r)}};
      tm.l(K + k - 1) = 1;
      tm.l(K + k - 2) = -1;
      out.push_back(tm);
    }
  }
  return out;
}

/**
 * Checks value, gradient, Hessian (fvar<var> and fvar<fvar<double>>), third
 * derivatives (fvar<fvar<var>>) and the derivatives of the value itself
 * (fvar<var> and fvar<fvar<double>>) of `lpmf(y, x, beta, cuts)` in
 * (beta, cuts).
 */
template <typename F, typename T_y, typename T_x>
void expect_derivatives(const F& lpmf, const T_y& y, const T_x& x,
                        const Eigen::VectorXd& beta,
                        const Eigen::VectorXd& cuts) {
  using stan::math::fvar;
  using stan::math::var;
  const int K = beta.size();
  const int M = K + cuts.size();
  Eigen::VectorXd v(M);
  v << beta, cuts;
  const double tiny = std::numeric_limits<double>::min();
  double lp = 0, lp_tol = 0;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(M), g_tol = g;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(M, M), H_tol = H;
  std::vector<Eigen::MatrixXd> T(M, H), T_tol(M, H);
  for (const auto& tm : terms(y, x, beta, cuts)) {
    Eigen::VectorXd a = tm.l.cwiseAbs();
    lp += tm.f[0];
    lp_tol += std::fabs(tm.f[0]);
    g += tm.f[1] * tm.l;
    g_tol += std::fabs(tm.f[1]) * a;
    H += tm.f[2] * tm.l * tm.l.transpose();
    H_tol += std::fabs(tm.f[2]) * a * a.transpose();
    for (int i = 0; i < M; ++i) {
      T[i] += tm.f[3] * tm.l(i) * tm.l * tm.l.transpose();
      T_tol[i] += std::fabs(tm.f[3]) * a(i) * a * a.transpose();
    }
  }

  auto f = [&](const auto& u) {
    return lpmf(y, x, u.head(K).eval(), u.tail(M - K).eval());
  };
  double fx;
  Eigen::VectorXd grad, grad_fwd;
  Eigen::MatrixXd H_fv, H_ffd, H_ffv;
  std::vector<Eigen::MatrixXd> grad_H;
  stan::math::hessian(f, v, fx, grad, H_fv);
  stan::math::hessian<double>(f, v, fx, grad_fwd, H_ffd);
  stan::math::grad_hessian(f, v, fx, H_ffv, grad_H);
  EXPECT_NEAR(fx, lp, 1e-13 * lp_tol + tiny);
  for (int i = 0; i < M; ++i) {
    const double gt = 1e-13 * g_tol(i) + tiny;
    EXPECT_NEAR(grad(i), g(i), gt) << i;
    EXPECT_NEAR(grad_fwd(i), g(i), gt) << i;
    for (int j = 0; j < M; ++j) {
      const double ht = 1e-13 * H_tol(i, j) + tiny;
      EXPECT_NEAR(H_fv(i, j), H(i, j), ht) << i << ", " << j;
      EXPECT_NEAR(H_ffd(i, j), H(i, j), ht) << i << ", " << j;
      EXPECT_NEAR(H_ffv(i, j), H(i, j), ht) << i << ", " << j;
      for (int k = 0; k < M; ++k) {
        EXPECT_NEAR(grad_H[i](j, k), T[i](j, k), 1e-13 * T_tol[i](j, k) + tiny)
            << i << ", " << j << ", " << k;
      }
    }
  }

  // derivative of the value itself
  stan::math::nested_rev_autodiff nested;
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> v_fv(M);
  for (int k = 0; k < M; ++k) {
    v_fv(k) = fvar<var>(v(k), 0);
  }
  f(v_fv).val_.grad();
  for (int i = 0; i < M; ++i) {
    const double gt = 1e-13 * g_tol(i) + tiny;
    EXPECT_NEAR(v_fv(i).val_.adj(), g(i), gt) << i;
    Eigen::Matrix<fvar<fvar<double>>, Eigen::Dynamic, 1> v_ff(M);
    for (int k = 0; k < M; ++k) {
      v_ff(k) = fvar<fvar<double>>(fvar<double>(v(k), i == k), 0);
    }
    EXPECT_NEAR(f(v_ff).val_.d_, g(i), gt) << i;
  }
}

/**
 * Runs `expect_derivatives` at locations equal to a cut, in the tails and
 * between two nearly equal cuts, for every class, vector and scalar y, and
 * (if `row_x`) a row-vector x.
 */
template <typename F>
void expect_all(const F& lpmf, bool row_x) {
  // dyadic x and beta: x_n beta = 2 x_n0 - x_n1 hits the dyadic cuts exactly
  Eigen::VectorXd beta(2);
  beta << 2, -1;
  Eigen::VectorXd cuts(3);
  cuts << -1, 0.5, 2;
  std::vector<int> y{1, 2, 2, 3, 3, 4, 1, 4, 2, 3, 1, 4, 2, 3};
  Eigen::VectorXd theta(14);
  theta << -1, -1, 0.5, 0.5, 2, 2, -40, 40, -40, 40, 750, -750, 0.25, 1.25;
  Eigen::MatrixXd x(14, 2);
  x.col(1) << 0.5, -1, 1, 0.25, -0.5, 2, 1.5, -0.75, 0.25, 0.125, -1, -0.25,
      0.75, 0.5;
  x.col(0) = (theta + x.col(1)) / 2;
  expect_derivatives(lpmf, y, x, beta, cuts);
  for (int k = 1; k <= 4; ++k) {
    expect_derivatives(lpmf, k, x, beta, cuts);
  }

  // cuts about 1e-9 apart; theta at, between and far from them (theta - cut
  // rounds at theta = 1000.2)
  const double gap = std::ldexp(1.0, -30) + std::ldexp(1.0, -50);
  Eigen::VectorXd close(3);
  close << -1, 0.5, 0.5 + gap;
  std::vector<int> y_close{3, 3, 3, 3, 3, 2, 4};
  Eigen::MatrixXd x_close = Eigen::MatrixXd::Zero(7, 2);
  x_close.col(0) << 0.25, 0.25 + gap / 4, 0.25 + gap / 2, 0.75, 500.1, 0.5,
      0.75;
  expect_derivatives(lpmf, y_close, x_close, beta, close);

  if (row_x) {
    for (int n : {0, 2, 6, 7, 10}) {
      Eigen::RowVectorXd x_row = x.row(n);
      expect_derivatives(lpmf, y, x_row, beta, cuts);
      for (int k = 1; k <= 4; ++k) {
        expect_derivatives(lpmf, k, x_row, beta, cuts);
      }
    }
  }
}

}  // namespace ordered_logistic_test

#endif

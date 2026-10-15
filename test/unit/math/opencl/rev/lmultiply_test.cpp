#ifdef STAN_OPENCL
#include <stan/math.hpp>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <limits>

auto lmultiply_functor
    = [](const auto& a, const auto& b) { return stan::math::lmultiply(a, b); };

TEST(OpenCLMatrix_lmultiply, prim_rev_values_small) {
  int N = 2;
  int M = 3;

  Eigen::MatrixXd a(N, M);
  a << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd b(N, M);
  b << 12, 11, 10, 9, 8, 7;
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, a, b);
}

TEST(OpenCLMatrixlmultiply, prim_rev_values_M_0) {
  int N = 2;
  int M = 0;

  Eigen::MatrixXd a(N, M);
  Eigen::MatrixXd b(N, M);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, a, b);

  Eigen::MatrixXd c(M, N);
  Eigen::MatrixXd d(M, N);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, c, d);
}

TEST(OpenCLMatrixlmultiply, prim_rev_values_large) {
  int N = 71;
  int M = 83;

  Eigen::MatrixXd a = Eigen::MatrixXd::Random(N, M);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(N, M);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, a, b);
}

TEST(OpenCLMatrixlmultiply, prim_rev_scalar_values_large) {
  int N = 71;
  int M = 83;

  Eigen::MatrixXd a = Eigen::MatrixXd::Random(N, M);
  double b = 0.3;
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, a, b);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, b, a);
}

// At a = b = 0 the partials are log(0) = -inf and 0 (see the rev test);
// the explicit values guard against CPU and OpenCL agreeing on NaN.
TEST(OpenCLMatrixlmultiply, prim_rev_derivative_at_zero) {
  using stan::math::from_matrix_cl;
  using stan::math::lmultiply;
  using stan::math::matrix_cl;
  using stan::math::sum;
  using stan::math::to_matrix_cl;
  using stan::math::var;
  using stan::math::var_value;
  constexpr double inf = std::numeric_limits<double>::infinity();
  Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(2, 3);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, zero, zero);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, zero, 0.0);
  stan::math::test::compare_cpu_opencl_prim_rev(lmultiply_functor, 0.0, zero);

  var_value<matrix_cl<double>> a(to_matrix_cl(zero));
  var_value<matrix_cl<double>> b(to_matrix_cl(zero));
  sum(lmultiply(a, b)).grad();
  EXPECT_MATRIX_EQ(Eigen::MatrixXd::Constant(2, 3, -inf),
                   from_matrix_cl(a.adj()));
  EXPECT_MATRIX_EQ(zero, from_matrix_cl(b.adj()));
  stan::math::recover_memory();

  var s = 0;
  var_value<matrix_cl<double>> c(to_matrix_cl(zero));
  sum(lmultiply(s, c)).grad();
  EXPECT_EQ(-inf, s.adj());
  EXPECT_MATRIX_EQ(zero, from_matrix_cl(c.adj()));

  // a result element with zero adjoint contributes 0 to a, not 0 * log(0)
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(2, 3);
  w(0, 0) = 1;
  var_value<matrix_cl<double>> d(to_matrix_cl(Eigen::MatrixXd::Ones(2, 3)));
  var_value<matrix_cl<double>> e(to_matrix_cl(zero));
  sum(stan::math::elt_multiply(to_matrix_cl(w), lmultiply(d, e))).grad();
  Eigen::MatrixXd d_adj = from_matrix_cl(d.adj());
  EXPECT_EQ(-inf, d_adj(0, 0));
  EXPECT_EQ(0, d_adj(1, 1));
  Eigen::MatrixXd e_adj = from_matrix_cl(e.adj());
  EXPECT_EQ(inf, e_adj(0, 0));
  EXPECT_EQ(0, e_adj(1, 1));
}
#endif

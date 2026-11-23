#ifdef STAN_OPENCL
#include <stan/math.hpp>
#include <test/unit/math/opencl/util.hpp>
#include <gtest/gtest.h>

// squared, so that the adjoint of the Kronecker product is not uniform
auto kronecker_functor = [](const auto& a, const auto& b) {
  auto k = stan::math::kronecker(a, b);
  return stan::math::eval(stan::math::elt_multiply(k, k));
};

TEST(OpenCLKronecker, prim_rev_values_small) {
  Eigen::MatrixXd a{{1, -2, 3}, {0.5, 4, -1}};
  Eigen::MatrixXd b{{2, 1}, {0, -3}, {5, 0.25}};
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, a, b);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, b, a);
}

TEST(OpenCLKronecker, prim_rev_size_0) {
  Eigen::MatrixXd a{{1, -2, 3}, {0.5, 4, -1}};
  Eigen::MatrixXd m00(0, 0);
  Eigen::MatrixXd m03(0, 3);
  Eigen::MatrixXd m20(2, 0);
  for (const auto& x : {a, m00, m03, m20}) {
    for (const auto& y : {a, m00, m03, m20}) {
      stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, x, y);
    }
  }
}

TEST(OpenCLKronecker, prim_rev_small_factor) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(40, 30);
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(1, 1);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, a, b);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, b, a);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, c, b);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, b, c);
}

TEST(OpenCLKronecker, prim_rev_colwise_partial_sums) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(1, 1);
  Eigen::MatrixXd r = Eigen::MatrixXd::Random(1, 2);
  Eigen::MatrixXd d = Eigen::MatrixXd::Random(100, 90);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, c, d);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, r, d);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, d, r);
}

TEST(OpenCLKronecker, prim_rev_values_large) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(7, 5);
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(4, 9);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, a, b);
  stan::math::test::compare_cpu_opencl_prim_rev(kronecker_functor, b, a);
}

#endif

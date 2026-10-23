#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixPrimMat, eigenvalues_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2, 3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_NO_THROW(eigenvalues_sym(m0));
  EXPECT_THROW(eigenvalues_sym(m1), std::invalid_argument);
}

TEST(MathMatrixPrimMat, eigenvalues_sym_hermitian) {
  using c = std::complex<double>;
  // Hermitian with trace 5 and determinant 4: eigenvalues 1 and 4
  Eigen::MatrixXcd m(2, 2);
  m << c(2, 0), c(1, -1), c(1, 1), c(3, 0);
  Eigen::VectorXcd ev = stan::math::eigenvalues_sym(m);
  ASSERT_EQ(2, ev.size());
  EXPECT_NEAR(1.0, ev(0).real(), 1e-12);
  EXPECT_NEAR(4.0, ev(1).real(), 1e-12);
  EXPECT_NEAR(0.0, ev(0).imag(), 1e-12);
  EXPECT_NEAR(0.0, ev(1).imag(), 1e-12);

  // symmetric but not Hermitian
  Eigen::MatrixXcd s(2, 2);
  s << c(1, 0), c(0, 1), c(0, 1), c(1, 0);
  EXPECT_THROW(stan::math::eigenvalues_sym(s), std::domain_error);
}

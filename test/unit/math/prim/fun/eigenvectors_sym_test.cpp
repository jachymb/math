#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixPrimMat, eigenvectors_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2, 3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d ev_m1(1, 1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_NO_THROW(eigenvectors_sym(m0));
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1), std::invalid_argument);
}

TEST(MathMatrixPrimMat, eigenvectors_sym_hermitian) {
  using c = std::complex<double>;
  // Hermitian with eigenvalues 1 and 4
  Eigen::MatrixXcd m(2, 2);
  m << c(2, 0), c(1, -1), c(1, 1), c(3, 0);
  Eigen::MatrixXcd v = stan::math::eigenvectors_sym(m);
  ASSERT_EQ(2, v.rows());
  ASSERT_EQ(2, v.cols());
  Eigen::Vector2d lambda(1.0, 4.0);
  for (Eigen::Index k = 0; k < 2; ++k) {
    EXPECT_NEAR(1.0, v.col(k).norm(), 1e-12);
    EXPECT_NEAR(0.0, (m * v.col(k) - lambda(k) * v.col(k)).norm(), 1e-12);
  }

  // symmetric but not Hermitian
  Eigen::MatrixXcd s(2, 2);
  s << c(1, 0), c(0, 1), c(0, 1), c(1, 0);
  EXPECT_THROW(stan::math::eigenvectors_sym(s), std::domain_error);
}

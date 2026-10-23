#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(Symmetrize_lower, Error) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::symmetrize_from_lower_tri;

  Matrix<double, Dynamic, Dynamic> m(3, 4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      m(i, j) = (i + 1) * (j + 1);
  EXPECT_THROW(symmetrize_from_lower_tri(m), std::invalid_argument);
}

TEST(Symmetrize_lower, Value) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::is_symmetric;
  using stan::math::symmetrize_from_lower_tri;

  Matrix<double, Dynamic, Dynamic> m(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      m(i, j) = (i + 2) * (j + 1);
  EXPECT_TRUE(is_symmetric(symmetrize_from_lower_tri(m)));
}

TEST(Symmetrize_lower, Value2) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using stan::math::is_symmetric;
  using stan::math::symmetrize_from_lower_tri;

  Matrix<double, Dynamic, Dynamic> m(4, 4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      m(i, j) = (i + 2) * (j + 1);
  MatrixXd m_sym_ref(4, 4);
  m_sym_ref << 2, 3, 4, 5, 3, 6, 8, 10, 4, 8, 12, 15, 5, 10, 15, 20;
  EXPECT_MATRIX_EQ(m_sym_ref, symmetrize_from_lower_tri(m));
}

TEST(Symmetrize_lower, ComplexHermitian) {
  using c = std::complex<double>;
  using stan::math::symmetrize_from_lower_tri;
  // the result is Hermitian: the lower triangle of m, its conjugate above
  // the diagonal, and the real part of the diagonal of m
  Eigen::MatrixXcd m(3, 3);
  m << c(1, 2), c(9, 9), c(9, 9), c(4, -3), c(0.5, 1), c(9, 9), c(2, 0.25),
      c(-0.5, -1.5), c(3, -1);
  Eigen::MatrixXcd s = symmetrize_from_lower_tri(m);
  for (int j = 0; j < 3; ++j) {
    EXPECT_EQ(c(m(j, j).real(), 0), s(j, j));
    for (int i = j + 1; i < 3; ++i) {
      EXPECT_EQ(m(i, j), s(i, j));
      EXPECT_EQ(std::conj(m(i, j)), s(j, i));
    }
  }
}

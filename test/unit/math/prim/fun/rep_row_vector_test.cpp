#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixPrimMat, rep_row_vector) {
  using stan::math::rep_row_vector;
  Eigen::Matrix<double, 1, Eigen::Dynamic> x = rep_row_vector(2.0, 3);
  EXPECT_EQ(3U, x.size());
  for (int i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(2.0, x[i]);
}

TEST(MathMatrixPrimMat, rep_row_vector_complex) {
  std::complex<double> z(1.5, -2.5);
  Eigen::RowVectorXcd x = stan::math::rep_row_vector(z, 3);
  EXPECT_EQ(3, x.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_EQ(z, x(i));
  }
}

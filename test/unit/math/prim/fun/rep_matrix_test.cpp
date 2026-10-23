#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixPrimMat, rep_matrix) {
  using stan::math::rep_matrix;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x
      = rep_matrix(2.0, 3, 4);
  EXPECT_EQ(12, x.size());
  EXPECT_EQ(3, x.rows());
  EXPECT_EQ(4, x.cols());
  for (int i = 0; i < x.rows(); ++i)
    for (int j = 0; j < x.cols(); ++j)
      EXPECT_FLOAT_EQ(2.0, x(i, j));

  EXPECT_THROW(rep_matrix(2.0, -1, 3), std::domain_error);
  EXPECT_THROW(rep_matrix(2.0, 3, -1), std::domain_error);
}

TEST(MathMatrixPrimMat, rep_matrix_always_returns_double) {
  int a = 1;
  auto x = stan::math::rep_matrix(a, 3, 4);

  EXPECT_TRUE((std::is_same<double, stan::value_type_t<decltype(x)>>::value));
}

TEST(MathMatrixPrimMat, rep_matrix_vec) {
  using stan::math::rep_matrix;
  Eigen::Matrix<double, Eigen::Dynamic, 1> v(3);
  v << 1.0, 4.0, 9.0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x = rep_matrix(v, 4);
  EXPECT_EQ(12, x.size());
  EXPECT_EQ(3, x.rows());
  EXPECT_EQ(4, x.cols());
  for (int i = 0; i < x.rows(); ++i)
    for (int j = 0; j < x.cols(); ++j)
      EXPECT_FLOAT_EQ(v(i), x(i, j));

  EXPECT_THROW(rep_matrix(v, -1), std::domain_error);
}

TEST(MathMatrixPrimMat, rep_matrix_row_vec) {
  using stan::math::rep_matrix;
  Eigen::Matrix<double, 1, Eigen::Dynamic> rv(3);
  rv << 1.0, 4.0, 9.0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x = rep_matrix(rv, 4);
  EXPECT_EQ(12, x.size());
  EXPECT_EQ(4, x.rows());
  EXPECT_EQ(3, x.cols());
  for (int i = 0; i < x.rows(); ++i)
    for (int j = 0; j < x.cols(); ++j)
      EXPECT_FLOAT_EQ(rv(j), x(i, j));

  EXPECT_THROW(rep_matrix(rv, -1), std::domain_error);
}

TEST(MathMatrixPrimMat, rep_matrix_both_ways) {
  using stan::math::rep_matrix;

  Eigen::Matrix<double, 1, Eigen::Dynamic> rv(3);
  rv << 1.0, 4.0, 9.0;

  Eigen::Matrix<double, Eigen::Dynamic, 1> v(3);
  v << 1.0, 4.0, 9.0;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> x(3, 3);
  x = true ? rep_matrix(v, 3) : rep_matrix(rv, 3);
}

TEST(MathMatrixPrimMat, rep_matrix_complex) {
  using c_t = std::complex<double>;
  using stan::math::rep_matrix;
  c_t z(1.5, -2.5);
  Eigen::MatrixXcd x = rep_matrix(z, 2, 3);
  EXPECT_EQ(2, x.rows());
  EXPECT_EQ(3, x.cols());
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.cols(); ++j) {
      EXPECT_EQ(z, x(i, j));
    }
  }

  Eigen::VectorXcd v(2);
  v << c_t(1, 2), c_t(-3, 4);
  Eigen::MatrixXcd y = rep_matrix(v, 3);
  EXPECT_EQ(2, y.rows());
  EXPECT_EQ(3, y.cols());
  Eigen::RowVectorXcd rv = v.transpose();
  Eigen::MatrixXcd w = rep_matrix(rv, 3);
  EXPECT_EQ(3, w.rows());
  EXPECT_EQ(2, w.cols());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(v(i), y(i, j));
      EXPECT_EQ(v(i), w(j, i));
    }
  }
}

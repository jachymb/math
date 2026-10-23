#include <stan/math/prim.hpp>
#include <gtest/gtest.h>

TEST(primMatFunRows, matrix) {
  using Eigen::MatrixXd;
  using stan::math::rows;
  MatrixXd m(3, 2);
  EXPECT_EQ(3, rows(m));

  MatrixXd m2(1, 2);
  EXPECT_EQ(1, rows(m2));
}
TEST(primMatFunRows, vector) {
  using Eigen::VectorXd;
  using stan::math::rows;
  VectorXd v(2);
  EXPECT_EQ(2, rows(v));

  VectorXd u(1);
  EXPECT_EQ(1, rows(u));
}
TEST(primMatFunRows, rowVector) {
  using Eigen::RowVectorXd;
  using stan::math::rows;
  RowVectorXd v(2);
  EXPECT_EQ(1, rows(v));

  RowVectorXd u(1);
  EXPECT_EQ(1, rows(u));
}
TEST(primMatFunRows, complex) {
  using stan::math::rows;
  Eigen::VectorXcd v(3);
  Eigen::RowVectorXcd rv(2);
  Eigen::MatrixXcd m(2, 3);
  EXPECT_EQ(3, rows(v));
  EXPECT_EQ(1, rows(rv));
  EXPECT_EQ(2, rows(m));
}

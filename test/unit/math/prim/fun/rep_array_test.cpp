#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(MathMatrixPrimArr, rep_array) {
  using stan::math::rep_array;
  std::vector<double> x = rep_array(2.0, 3);
  EXPECT_EQ(3U, x.size());
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(2.0, x[i]);

  EXPECT_THROW(rep_array(2.0, -2), std::domain_error);
}
TEST(MathMatrixPrimArr, rep_array2D) {
  using stan::math::rep_array;
  using std::vector;
  vector<vector<double> > x = rep_array(2.0, 3, 4);
  EXPECT_EQ(3U, x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(4U, x[i].size());
    for (size_t j = 0; j < x[i].size(); ++j)
      EXPECT_FLOAT_EQ(2.0, x[i][j]);
  }
  EXPECT_THROW(rep_array(2.0, -2, 3), std::domain_error);
  EXPECT_THROW(rep_array(2.0, 2, -3), std::domain_error);
}
TEST(MathMatrixPrimArr, rep_array3D) {
  using stan::math::rep_array;
  using std::vector;
  vector<vector<vector<int> > > x = rep_array(13, 3, 4, 5);
  EXPECT_EQ(3U, x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(4U, x[i].size());
    for (size_t j = 0; j < x[i].size(); ++j) {
      EXPECT_EQ(5U, x[i][j].size());
      for (size_t k = 0; k < x[i][j].size(); ++k)
        EXPECT_EQ(13, x[i][j][k]);
    }
  }
  EXPECT_THROW(rep_array(2.0, -2, 3, 4), std::domain_error);
  EXPECT_THROW(rep_array(2.0, 2, -3, 4), std::domain_error);
  EXPECT_THROW(rep_array(2.0, 2, 3, -4), std::domain_error);
}
TEST(MathMatrixPrimArr, rep_array_complex) {
  using c_t = std::complex<double>;
  using stan::math::rep_array;
  using std::vector;
  c_t z(1.5, -2.5);
  vector<vector<vector<c_t> > > x = rep_array(z, 2, 3, 4);
  EXPECT_EQ(2U, x.size());
  for (const auto& x_i : x) {
    EXPECT_EQ(3U, x_i.size());
    for (const auto& x_ij : x_i) {
      EXPECT_EQ(4U, x_ij.size());
      for (const auto& x_ijk : x_ij) {
        EXPECT_EQ(z, x_ijk);
      }
    }
  }

  Eigen::VectorXcd v(2);
  v << c_t(1, 2), c_t(-3, 4);
  vector<Eigen::VectorXcd> y = rep_array(v, 3);
  EXPECT_EQ(3U, y.size());
  for (const auto& y_i : y) {
    ASSERT_EQ(2, y_i.size());
    EXPECT_EQ(v(0), y_i(0));
    EXPECT_EQ(v(1), y_i(1));
  }
}

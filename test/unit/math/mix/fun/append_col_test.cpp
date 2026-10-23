#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, appendCol) {
  auto f = [](const auto& x, const auto& y) {
    return stan::math::append_col(x, y);
  };

  Eigen::MatrixXd m22(2, 2);
  m22 << 2, 3, 9, -1;
  Eigen::MatrixXd m22b(2, 2);
  m22b << 4, 3, 0, 1;
  Eigen::MatrixXd m23b(2, 3);
  m23b << 4, 3, 0, 1, 11, 12;
  Eigen::VectorXd v2(2);
  v2 << 17, -32;
  Eigen::VectorXd v2b(2);
  v2b << 11, 9;
  Eigen::RowVectorXd rv4(4);
  rv4 << 2, 3, 9, -1;
  Eigen::RowVectorXd rv3(3);
  rv3 << 4, 3, 0.4;

  // (matrix, matrix)
  stan::test::expect_ad(f, m22, m22b);
  stan::test::expect_ad(f, m22, m23b);
  stan::test::expect_ad_matvar(f, m22, m22b);
  stan::test::expect_ad_matvar(f, m22, m23b);

  // (vector, matrix)
  stan::test::expect_ad(f, v2, m22);
  stan::test::expect_ad_matvar(f, v2, m22);

  // (matrix, vector)
  stan::test::expect_ad(f, m22, v2);
  stan::test::expect_ad_matvar(f, m22, v2);

  // (vector, vector)
  stan::test::expect_ad(f, v2, v2b);
  stan::test::expect_ad_matvar(f, v2, v2b);

  // (row vector, row vector)
  stan::test::expect_ad(f, rv3, rv4);
  stan::test::expect_ad(f, rv4, rv3);
  stan::test::expect_ad_matvar(f, rv3, rv4);
  stan::test::expect_ad_matvar(f, rv4, rv3);

  Eigen::MatrixXd m00(0, 0);
  Eigen::MatrixXd m03(0, 3);
  Eigen::VectorXd v0(0);
  Eigen::RowVectorXd rv0(0);

  stan::test::expect_ad(f, m00, m00);
  stan::test::expect_ad(f, m00, m03);
  stan::test::expect_ad(f, v0, m00);
  stan::test::expect_ad(f, v0, v0);
  stan::test::expect_ad(f, rv0, rv0);

  stan::test::expect_ad_matvar(f, m00, m00);
  stan::test::expect_ad_matvar(f, m00, m03);
  stan::test::expect_ad_matvar(f, v0, m00);
  stan::test::expect_ad_matvar(f, v0, v0);
  stan::test::expect_ad_matvar(f, rv0, rv0);
}

TEST(MathMixMatFun, appendColComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x, const auto& y) {
    return stan::math::append_col(x, y);
  };

  c_t z(0.4, 2.9);
  Eigen::MatrixXcd m22(2, 2);
  m22 << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9);
  Eigen::MatrixXcd m23(2, 3);
  m23 << c_t(0.2, -1.1), c_t(1.7, 0.6), c_t(-0.3, -2.4), c_t(2.1, 1.3),
      c_t(-1.8, 0.5), c_t(0.9, -0.7);
  Eigen::VectorXcd v2(2);
  v2 << c_t(1.5, -2.1), c_t(-3.2, 0.7);
  Eigen::VectorXcd v2b(2);
  v2b << c_t(0.6, 1.2), c_t(-1.7, -0.4);
  Eigen::RowVectorXcd rv2(2);
  rv2 << c_t(2.3, -0.8), c_t(-1.9, 1.6);
  Eigen::RowVectorXcd rv3(3);
  rv3 << c_t(-0.2, 0.9), c_t(1.4, -1.5), c_t(0.7, 0.1);

  stan::test::expect_ad(f, z, rv2);
  stan::test::expect_ad(f, rv2, z);
  stan::test::expect_ad(f, rv2, rv3);
  stan::test::expect_ad(f, v2, v2b);
  stan::test::expect_ad(f, v2, m22);
  stan::test::expect_ad(f, m22, v2);
  stan::test::expect_ad(f, m22, m23);
}

#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, elt_divide_transpose_test) {
  auto f
      = [](const auto& x) { return stan::math::elt_divide(x, x.transpose()); };

  Eigen::MatrixXd x(2, 2);

  stan::test::expect_ad_matvar(f, x);
}

TEST(MathMixMatFun, elt_divide_block_test) {
  auto f = [](const auto& y, const auto& x) {
    return stan::math::elt_divide(y.block(1, 1, 2, 2), x.block(0, 0, 2, 2));
  };

  Eigen::MatrixXd x(3, 3);

  stan::test::expect_ad_matvar(f, x, x);
}

TEST(MathMixMatFun, eltDivide) {
  auto f = [](const auto& x, const auto& y) {
    return stan::math::elt_divide(x, y);
  };

  Eigen::VectorXd a2(2);
  a2 << 2, 5;
  Eigen::VectorXd b2(2);
  b2 << 10, 100;
  stan::test::expect_ad(f, a2, b2);
  stan::test::expect_ad_matvar(f, a2, b2);

  Eigen::RowVectorXd c2(2);
  c2 << 2, 5;
  Eigen::RowVectorXd d2(2);
  d2 << 10, 100;
  stan::test::expect_ad(f, c2, d2);
  stan::test::expect_ad_matvar(f, c2, d2);

  Eigen::MatrixXd e23(2, 3);
  e23 << 2, 5, 7, 13, 29, 112;
  Eigen::MatrixXd f23(2, 3);
  f23 << 1e1, 1e2, 1e3, 1e4, 1e5, 1e6;
  stan::test::expect_ad(f, e23, f23);
  stan::test::expect_ad_matvar(f, e23, f23);

  double z = 10;
  stan::test::expect_ad(f, a2, z);
  stan::test::expect_ad(f, c2, z);
  stan::test::expect_ad(f, e23, z);

  stan::test::expect_ad_matvar(f, a2, z);
  stan::test::expect_ad_matvar(f, c2, z);
  stan::test::expect_ad_matvar(f, e23, z);

  stan::test::expect_ad(f, z, a2);
  stan::test::expect_ad(f, z, c2);
  stan::test::expect_ad(f, z, e23);

  stan::test::expect_ad_matvar(f, z, a2);
  stan::test::expect_ad_matvar(f, z, c2);
  stan::test::expect_ad_matvar(f, z, e23);

  Eigen::VectorXd v0(0);
  Eigen::RowVectorXd rv0(0);
  Eigen::MatrixXd m00(0, 0);
  stan::test::expect_ad(f, v0, v0);
  stan::test::expect_ad(f, rv0, rv0);
  stan::test::expect_ad(f, m00, m00);
  stan::test::expect_ad(f, v0, z);
  stan::test::expect_ad(f, rv0, z);
  stan::test::expect_ad(f, m00, z);

  stan::test::expect_ad_matvar(f, v0, v0);
  stan::test::expect_ad_matvar(f, rv0, rv0);
  stan::test::expect_ad_matvar(f, m00, m00);
  stan::test::expect_ad_matvar(f, v0, z);
  stan::test::expect_ad_matvar(f, rv0, z);
  stan::test::expect_ad_matvar(f, m00, z);
}

TEST(MathMixMatFun, eltDivideComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x, const auto& y) {
    return stan::math::elt_divide(x, y);
  };

  c_t z1(1.5, -0.5);
  c_t z2(-0.8, 1.2);
  Eigen::VectorXcd v1(2);
  v1 << c_t(0.5, 1.2), c_t(-1.3, 0.4);
  Eigen::VectorXcd v2(2);
  v2 << c_t(2.1, -0.7), c_t(0.3, 1.9);
  Eigen::RowVectorXcd rv1(2);
  rv1 << c_t(1.1, -0.6), c_t(-0.4, 2.2);
  Eigen::RowVectorXcd rv2(2);
  rv2 << c_t(0.7, 0.9), c_t(-1.6, -0.3);
  Eigen::MatrixXcd m1(2, 2);
  m1 << c_t(1, -0.5), c_t(0.25, 2), c_t(-1.5, 0.75), c_t(2.5, -1);
  Eigen::MatrixXcd m2(2, 2);
  m2 << c_t(0.6, 1.4), c_t(-2, 0.3), c_t(1.2, -0.9), c_t(-0.7, -1.8);

  stan::test::expect_ad(f, z1, z2);
  stan::test::expect_ad(f, z1, v1);
  stan::test::expect_ad(f, z1, rv1);
  stan::test::expect_ad(f, z1, m1);
  stan::test::expect_ad(f, v1, z2);
  stan::test::expect_ad(f, v1, v2);
  stan::test::expect_ad(f, rv1, z2);
  stan::test::expect_ad(f, rv1, rv2);
  stan::test::expect_ad(f, m1, z2);
  stan::test::expect_ad(f, m1, m2);
}

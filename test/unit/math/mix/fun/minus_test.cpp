#include <test/unit/math/test_ad.hpp>
#include <vector>

TEST(MathMixMatFun, minus) {
  auto f = [](const auto& x) { return stan::math::minus(x); };

  double x = 10;
  stan::test::expect_ad(f, x);

  Eigen::VectorXd y0(0);
  Eigen::VectorXd y1(1);
  y1 << -100;
  Eigen::VectorXd y2(2);
  y2 << -100, 0;
  Eigen::VectorXd y3(3);
  y3 << -100, 0, 1;
  for (const auto& y : std::vector<Eigen::VectorXd>{y0, y1, y2, y3}) {
    stan::test::expect_ad(f, y);
  }

  Eigen::RowVectorXd z0(0);
  Eigen::RowVectorXd z1(1);
  z1 << -100;
  Eigen::RowVectorXd z2(2);
  z2 << -100, 0;
  Eigen::RowVectorXd z3(3);
  z3 << -100, 0, 1;
  for (const auto& y : std::vector<Eigen::RowVectorXd>{z0, z1, z2, z3}) {
    stan::test::expect_ad(f, y);
  }

  Eigen::MatrixXd u00(0, 0);
  Eigen::MatrixXd u11(1, 1);
  u11 << 1;
  Eigen::MatrixXd u22(2, 2);
  u22 << 1, 2, 3, 4;
  Eigen::MatrixXd u23(2, 3);
  u23 << -100, 0, 1, 20, -40, 2;
  for (const auto& y : std::vector<Eigen::MatrixXd>{u00, u11, u22, u23}) {
    stan::test::expect_ad(f, y);
  }

  Eigen::MatrixXd v22(2, 2);
  v22 << 5, 6, 7, 8;
  std::vector<Eigen::MatrixXd> vec_mat{u22, v22};
  stan::test::expect_ad(f, vec_mat);
}

TEST(MathMixMatFun, minusComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::minus(x); };

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

  stan::test::expect_ad(f, z1);
  stan::test::expect_ad(f, v1);
  stan::test::expect_ad(f, rv1);
  stan::test::expect_ad(f, m1);
  stan::test::expect_ad(f, std::vector<c_t>{z1, z2});
  stan::test::expect_ad(f, std::vector<Eigen::VectorXcd>{v1, v2});
  stan::test::expect_ad(f, std::vector<Eigen::RowVectorXcd>{rv1, rv2});
  stan::test::expect_ad(f, std::vector<Eigen::MatrixXcd>{m1, m2});
  stan::test::expect_ad(f, std::vector<std::vector<c_t>>{{z1, z2}, {z2, z1}});
}

#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, diagonal) {
  auto f = [](const auto& x) { return stan::math::diagonal(x); };

  Eigen::MatrixXd a0(0, 0);
  stan::test::expect_ad(f, a0);

  Eigen::MatrixXd a1(1, 1);
  a1 << 1;
  stan::test::expect_ad(f, a1);

  Eigen::MatrixXd a2(2, 2);
  a2 << 1, 2, 3, 4;
  stan::test::expect_ad(f, a2);

  Eigen::MatrixXd a3(3, 3);
  a3 << 1, 4, 9, 1, 4, 9, 1, 4, 9;
  stan::test::expect_ad(f, a3);

  Eigen::MatrixXd a32(3, 2);
  a32 << 1, 2, 3, 4, 5, 6;
  stan::test::expect_ad(f, a32);

  Eigen::MatrixXd a23(2, 3);
  a23 << 1, 2, 3, 4, 5, 6;
  stan::test::expect_ad(f, a23);
}

TEST(MathMixMatFun, diagonalComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::diagonal(x); };
  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9), c_t(2.2, 1.1),
      c_t(-0.6, -1.8), c_t(3.3, -0.9);
  stan::test::expect_ad(f, m);
}

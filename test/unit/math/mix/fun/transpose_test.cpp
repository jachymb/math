#include <test/unit/math/test_ad.hpp>

inline void expect_transpose(const Eigen::MatrixXd& m) {
  auto f = [](const auto& x) { return stan::math::transpose(x); };
  Eigen::VectorXd v = stan::test::to_vector(m);
  Eigen::RowVectorXd rv = stan::test::to_row_vector(m);
  stan::test::expect_ad(f, m);
  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
}

TEST(MathMixMatFun, transpose) {
  Eigen::MatrixXd a(0, 0);
  expect_transpose(a);

  Eigen::MatrixXd b(1, 1);
  b << -1.2;
  expect_transpose(b);

  Eigen::MatrixXd c(2, 2);
  c << -1, 2, 5, 10;
  expect_transpose(c);

  Eigen::MatrixXd d(2, 3);
  d << -1, 2, -3, 5, 10, 100;
  expect_transpose(d);

  Eigen::MatrixXd e(3, 1);
  e << 1, 2, 3;
  expect_transpose(e);

  Eigen::MatrixXd g(3, 3);
  g << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  expect_transpose(g);
}

TEST(MathMixMatFun, transpose_complex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::transpose(x); };

  Eigen::VectorXcd v(3);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9);
  Eigen::RowVectorXcd rv = v.transpose();
  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9),
      c_t(2.4, 1.7), c_t(-1.9, -3.3);

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, m);
}

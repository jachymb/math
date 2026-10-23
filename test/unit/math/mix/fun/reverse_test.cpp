#include <test/unit/math/test_ad.hpp>
#include <vector>

TEST(MathMixMatFun, reverse_vector) {
  auto f = [](const auto& x) { return stan::math::reverse(x); };

  // 0 x 0
  Eigen::VectorXd x0(0);
  stan::test::expect_ad(f, x0);

  // 1 x 1
  Eigen::VectorXd x1(1);
  x1 << 1;
  stan::test::expect_ad(f, x1);

  // 4 x 4
  Eigen::VectorXd x(4);
  x << 1, 3, 4, -5;
  stan::test::expect_ad(f, x);
}

TEST(MathMixMatFun, reverse_row_vector) {
  auto f = [](const auto& x) { return stan::math::reverse(x); };

  // 0 x 0
  Eigen::RowVectorXd x0(0);
  stan::test::expect_ad(f, x0);

  // 1 x 1
  Eigen::RowVectorXd x1(1);
  x1 << 1;
  stan::test::expect_ad(f, x1);

  // 4 x 4
  Eigen::RowVectorXd x(4);
  x << 1, 3, 4, -5;
  stan::test::expect_ad(f, x);
}

TEST(MathMixMatFun, reverse_array) {
  auto f = [](const auto& x) { return stan::math::reverse(x); };

  // 0 x 0
  std::vector<double> x0(0);
  stan::test::expect_ad(f, x0);

  // 1 x 1
  std::vector<double> x1{1};
  stan::test::expect_ad(f, x1);

  // 4 x 4
  std::vector<double> x{1, 3, 4, -5};
  stan::test::expect_ad(f, x);
}

TEST(MathMixMatFun, reverse_complex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::reverse(x); };

  Eigen::VectorXcd v(3);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9);
  Eigen::RowVectorXcd rv = v.transpose();
  Eigen::MatrixXcd m(2, 2);
  m << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9);
  std::vector<c_t> a{v(0), v(1), v(2)};
  std::vector<Eigen::VectorXcd> av{v, v.reverse(), v.conjugate()};
  std::vector<Eigen::RowVectorXcd> arv{rv, rv.reverse(), rv.conjugate()};
  std::vector<Eigen::MatrixXcd> am{m, m.transpose(), m.conjugate()};
  std::vector<std::vector<c_t>> aa{a, {v(2), v(0), v(1)}, {v(1), v(2), v(0)}};

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, a);
  stan::test::expect_ad(f, av);
  stan::test::expect_ad(f, arv);
  stan::test::expect_ad(f, am);
  stan::test::expect_ad(f, aa);
}

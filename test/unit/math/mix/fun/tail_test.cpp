#include <test/unit/math/test_ad.hpp>
#include <vector>

template <typename T>
inline void expect_tail(const T& x, int n) {
  auto f = [](int i) {
    return [=](const auto& y) { return stan::math::tail(y, i); };
  };
  Eigen::VectorXd v = stan::test::to_vector(x);
  Eigen::RowVectorXd rv = stan::test::to_row_vector(x);
  stan::test::expect_ad(f(n), x);
  stan::test::expect_ad(f(n), v);
  stan::test::expect_ad(f(n), rv);
}

TEST(MathMixMatFun, tail) {
  std::vector<double> a{};
  expect_tail(a, 0);
  expect_tail(a, 1);

  std::vector<double> b{1};
  expect_tail(b, 0);
  expect_tail(b, 1);
  expect_tail(b, 2);

  std::vector<double> v{1, 2, 3};
  for (int n = 0; n < 5; ++n) {
    expect_tail(v, n);
  }
}

TEST(MathMixMatFun, tailEig) {
  Eigen::VectorXd a(0);
  expect_tail(a, 0);
  expect_tail(a, 1);

  Eigen::VectorXd b(1);
  b << 1;
  expect_tail(b, 0);
  expect_tail(b, 1);
  expect_tail(b, 2);

  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  for (int n = 0; n < 5; ++n) {
    expect_tail(v, n);
  }
}

TEST(MathMixMatFun, tailComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& y) { return stan::math::tail(y, 2); };

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

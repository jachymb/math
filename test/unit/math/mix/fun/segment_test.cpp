#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, segment) {
  auto f = [](int i, int n) {
    return [=](const auto& y) { return stan::math::segment(y, i, n); };
  };

  Eigen::VectorXd a(0);
  stan::test::expect_ad(f(1, 0), a);  // out of bounds
  stan::test::expect_ad(f(1, 2), a);  // out of bounds

  Eigen::VectorXd b(1);
  b << 1.2;
  stan::test::expect_ad(f(1, 1), b);
  stan::test::expect_ad(f(1, 2), b);  // out of bounds

  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  Eigen::RowVectorXd rv = v;
  stan::test::expect_ad(f(1, 0), v);
  stan::test::expect_ad(f(1, 3), v);
  stan::test::expect_ad(f(1, 0), rv);
  stan::test::expect_ad(f(1, 3), rv);

  // exception (out of bounds)
  stan::test::expect_ad(f(1, 4), v);   // out of bounds
  stan::test::expect_ad(f(1, 4), rv);  // out of bounds
  stan::test::expect_ad(f(1, 7), v);   // out of bounds
  stan::test::expect_ad(f(1, 7), rv);  // out of bounds

  Eigen::VectorXd u(4);
  u << 1, 2, 3, 4;
  Eigen::RowVectorXd ru = u;
  stan::test::expect_ad(f(2, 2), ru);
}

TEST(MathMixMatFun, segmentComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& y) { return stan::math::segment(y, 2, 2); };

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

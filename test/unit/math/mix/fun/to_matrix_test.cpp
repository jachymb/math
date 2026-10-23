#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, to_matrix_complex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::to_matrix(x); };
  auto g = [](int r, int c) {
    return [=](const auto& x) { return stan::math::to_matrix(x, r, c); };
  };
  // col_major = 0: row-major fill
  auto h = [](int r, int c) {
    return [=](const auto& x) { return stan::math::to_matrix(x, r, c, 0); };
  };

  Eigen::VectorXcd v(4);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9), c_t(2.3, -1.6);
  Eigen::RowVectorXcd rv = v.transpose();
  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9),
      c_t(2.4, 1.7), c_t(-1.9, -3.3);
  std::vector<c_t> a{v(0), v(1), v(2), v(3)};
  std::vector<Eigen::VectorXcd> av{v.head(2), v.tail(2)};
  std::vector<Eigen::RowVectorXcd> arv{rv.head(2), rv.tail(2)};
  std::vector<std::vector<c_t>> aa{{v(0), v(1)}, {v(2), v(3)}};

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, m);
  stan::test::expect_ad(f, av);
  stan::test::expect_ad(f, arv);
  stan::test::expect_ad(f, aa);

  stan::test::expect_ad(g(2, 2), v);
  stan::test::expect_ad(g(2, 2), rv);
  stan::test::expect_ad(g(3, 2), m);
  stan::test::expect_ad(g(2, 2), a);

  stan::test::expect_ad(h(2, 2), v);
  stan::test::expect_ad(h(2, 2), rv);
  stan::test::expect_ad(h(3, 2), m);
  stan::test::expect_ad(h(2, 2), a);
}

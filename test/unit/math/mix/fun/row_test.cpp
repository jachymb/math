#include <test/unit/math/test_ad.hpp>
#include <vector>

TEST(MathMixMatFun, row) {
  auto f = [](int i) {
    return [=](const auto& y) { return stan::math::row(y, i); };
  };

  // normal and exception returns
  Eigen::MatrixXd a(0, 0);

  Eigen::MatrixXd b(1, 1);
  b << 1;

  Eigen::MatrixXd c(2, 2);
  c << 1, 2, 3, 4;

  Eigen::MatrixXd d(3, 2);
  d << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXd e(2, 3);
  e << 1, 2, 3, 4, 5, 6;

  for (const auto& x : std::vector<Eigen::MatrixXd>{a, b, c, d, e})
    for (size_t i = 0; i < 5; ++i)
      stan::test::expect_ad(f(i), x);
}

TEST(MathMixMatFun, rowComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& y) { return stan::math::row(y, 2); };
  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9), c_t(2.2, 1.1),
      c_t(-0.6, -1.8), c_t(3.3, -0.9);
  stan::test::expect_ad(f, m);
}

#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, symmetrize_from_lower_tri) {
  auto f
      = [](const auto& x) { return stan::math::symmetrize_from_lower_tri(x); };

  Eigen::MatrixXd a(0, 0);
  stan::test::expect_ad(f, a);

  Eigen::MatrixXd b(1, 1);
  b << -1.2;
  stan::test::expect_ad(f, b);

  Eigen::MatrixXd c(2, 2);
  c << -1, 2, 5, 10;
  stan::test::expect_ad(f, c);

  // Test invalid sizes
  Eigen::MatrixXd d(2, 3);
  d << -1, 2, -3, 5, 10, 100;
  stan::test::expect_ad(f, d);
}

TEST(MathMixMatFun, symmetrize_from_lower_tri_complex) {
  using cd = std::complex<double>;
  auto f
      = [](const auto& x) { return stan::math::symmetrize_from_lower_tri(x); };

  Eigen::MatrixXcd m(3, 3);
  m << cd(1, 2), cd(2, -1), cd(-3, 0.5), cd(4, -3), cd(0.5, 1), cd(-1, 2),
      cd(2, 0.25), cd(-0.5, -1.5), cd(3, 1);
  stan::test::expect_ad(f, m);
}

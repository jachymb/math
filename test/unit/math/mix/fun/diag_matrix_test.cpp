#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, diagMatrix) {
  auto f = [](const auto& y) { return stan::math::diag_matrix(y); };

  for (int i = 0; i < 5; ++i) {
    Eigen::VectorXd a(i);
    for (int n = 0; n < a.size(); ++n)
      a(n) = n;
    stan::test::expect_ad(f, a);
    Eigen::RowVectorXd b = a.transpose();
    stan::test::expect_ad(f, b);
  }
}

TEST(MathMixMatFun, diagMatrixComplex) {
  using cd = std::complex<double>;
  auto f = [](const auto& y) { return stan::math::diag_matrix(y); };

  Eigen::VectorXcd a(3);
  a << cd(1, 2), cd(-0.5, 1.5), cd(3, -1);
  stan::test::expect_ad(f, a);
}

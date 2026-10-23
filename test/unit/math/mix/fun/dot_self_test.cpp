#include <test/unit/math/test_ad.hpp>
#include <vector>

TEST(MathMixMatFun, dotSelf) {
  auto f = [](const auto& y) { return stan::math::dot_self(y); };

  Eigen::VectorXd x0(0);

  Eigen::VectorXd x1(1);
  x1 << 2;

  Eigen::VectorXd x2(2);
  x2 << 2, 3;

  Eigen::VectorXd x3(3);
  x3 << 2, 3, 4;

  for (const auto& a : std::vector<Eigen::VectorXd>{x0, x1, x2, x3}) {
    stan::test::expect_ad(f, a);
    stan::test::expect_ad_matvar(f, a);
  }
}

TEST(MathMixMatFun, dotSelfComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& y) { return stan::math::dot_self(y); };

  Eigen::VectorXcd v(3);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9);
  Eigen::RowVectorXcd rv = v.transpose();

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
}

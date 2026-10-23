#include <test/unit/math/test_ad.hpp>
#include <vector>

template <typename T>
inline void expect_prod(const T& m) {
  auto f = [](const auto& x) { return stan::math::prod(x); };
  Eigen::VectorXd v(m.size());
  for (int i = 0; i < m.size(); ++i) {
    v(i) = m(i);
  }
  Eigen::RowVectorXd rv = v;
  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, m);
}

TEST(mathMixMatFun, prod) {
  Eigen::MatrixXd m00(0, 0);
  Eigen::MatrixXd m11(1, 1);
  m11 << 2;
  Eigen::MatrixXd m22(2, 2);
  m22 << 2, 3, 2, 3;
  Eigen::MatrixXd m23(2, 3);
  m23 << 10, 20, 30, -10, -30, -100;
  for (const auto& m : std::vector<Eigen::MatrixXd>{m00, m11, m22, m23}) {
    expect_prod(m);
  }
}

TEST(mathMixMatFun, prodComplex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::prod(x); };

  Eigen::VectorXcd v(3);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9);
  Eigen::RowVectorXcd rv = v.transpose();
  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.1, 0.3), c_t(-0.5, 1.2), c_t(0.8, -1.4), c_t(-1.3, -0.9),
      c_t(0.6, 0.7), c_t(1.2, -0.4);
  std::vector<c_t> a{v(0), v(1), v(2)};

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, m);
  stan::test::expect_ad(f, a);
}

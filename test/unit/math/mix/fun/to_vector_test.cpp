#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, to_vector_complex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::to_vector(x); };

  Eigen::VectorXcd v(3);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7), c_t(0.4, 2.9);
  Eigen::RowVectorXcd rv = v.transpose();
  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9),
      c_t(2.4, 1.7), c_t(-1.9, -3.3);
  std::vector<c_t> a{v(0), v(1), v(2)};

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, m);
  stan::test::expect_ad(f, a);
}

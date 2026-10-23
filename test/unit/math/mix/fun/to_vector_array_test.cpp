#include <test/unit/math/test_ad.hpp>

TEST(MathMixMatFun, to_vector_array_complex) {
  using c_t = std::complex<double>;
  auto f = [](const auto& x) { return stan::math::to_vector_array(x); };

  Eigen::MatrixXcd m(2, 3);
  m << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9),
      c_t(2.4, 1.7), c_t(-1.9, -3.3);

  stan::test::expect_ad(f, m);
}

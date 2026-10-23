#include <test/unit/math/test_ad.hpp>
#include <complex>

TEST(mathMixMatFun, get_real) {
  auto f = [](const auto& z) { return stan::math::get_real(z); };
  stan::test::expect_complex_common(f);
}

template <typename T>
inline void test_vectorized_get_real() {
  using complex_t = std::complex<T>;
  using matrix_t = Eigen::Matrix<T, -1, -1>;
  using complex_matrix = Eigen::Matrix<complex_t, -1, -1>;
  complex_matrix A(2, 2);
  A << complex_t(T(0), T(1)), complex_t(T(2), T(3)), complex_t(T(4), T(5)),
      complex_t(T(6), T(7));
  matrix_t values = A.real();
  matrix_t A_imag = stan::math::get_real(A);
  EXPECT_MATRIX_EQ(A_imag, values);
  std::vector<complex_matrix> A_vec{A, A, A};
  std::vector<matrix_t> A_vec_imag = stan::math::get_real(A_vec);
  for (auto&& a_imag : A_vec_imag) {
    EXPECT_MATRIX_EQ(a_imag, values);
  }
  std::vector<std::vector<complex_matrix>> A_vec_vec{A_vec, A_vec, A_vec};
  std::vector<std::vector<matrix_t>> A_vec_vec_imag
      = stan::math::get_real(A_vec_vec);
  for (auto&& a_vec_imag : A_vec_vec_imag) {
    for (auto&& a_imag : a_vec_imag) {
      EXPECT_MATRIX_EQ(a_imag, values);
    }
  }
}
TEST(mathMixMatFun, get_real_vectorized) {
  test_vectorized_get_real<double>();
  test_vectorized_get_real<stan::math::var>();
  test_vectorized_get_real<stan::math::fvar<double>>();
  test_vectorized_get_real<stan::math::fvar<stan::math::var>>();
  test_vectorized_get_real<
      stan::math::fvar<stan::math::fvar<stan::math::var>>>();
}

TEST(mathMixMatFun, get_real_containers) {
  auto f = [](const auto& z) { return stan::math::get_real(z); };
  using c_t = std::complex<double>;
  Eigen::VectorXcd v(3);
  v << c_t{1.5, -0.7}, c_t{-2.1, 0.4}, c_t{0.3, 3.2};
  Eigen::RowVectorXcd rv(3);
  rv << c_t{-1.2, -0.9}, c_t{2.4, 1.1}, c_t{-0.6, 2.7};
  Eigen::MatrixXcd m(2, 3);
  m << c_t{1.5, -0.7}, c_t{-2.1, 0.4}, c_t{0.3, 3.2}, c_t{-1.2, -0.9},
      c_t{2.4, 1.1}, c_t{-0.6, 2.7};
  std::vector<c_t> a{c_t{1.5, -0.7}, c_t{-2.1, 0.4}};
  std::vector<c_t> b{c_t{0.3, 3.2}, c_t{-1.2, -0.9}};

  stan::test::expect_ad(f, v);
  stan::test::expect_ad(f, rv);
  stan::test::expect_ad(f, m);
  stan::test::expect_ad(f, a);
  stan::test::expect_ad(f, std::vector<Eigen::VectorXcd>{v, v});
  stan::test::expect_ad(f, std::vector<Eigen::RowVectorXcd>{rv, rv});
  stan::test::expect_ad(f, std::vector<Eigen::MatrixXcd>{m, m});
  stan::test::expect_ad(f, std::vector<std::vector<c_t>>{a, b});
}

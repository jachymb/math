#include <test/unit/math/test_ad.hpp>
#include <vector>

TEST(MathMixMatFun, repArrayComplex) {
  using c_t = std::complex<double>;
  auto f1 = [](const auto& x) { return stan::math::rep_array(x, 2); };
  auto f2 = [](const auto& x) { return stan::math::rep_array(x, 2, 1); };
  auto f3 = [](const auto& x) { return stan::math::rep_array(x, 1, 2, 1); };
  auto expect_rep = [f1, f2, f3](const auto& x) {
    stan::test::expect_ad(f1, x);
    stan::test::expect_ad(f2, x);
    stan::test::expect_ad(f3, x);
  };

  c_t z(1.5, -2.1);
  Eigen::VectorXcd v(2);
  v << c_t(1.5, -2.1), c_t(-3.2, 0.7);
  Eigen::RowVectorXcd rv(2);
  rv << c_t(0.4, 2.9), c_t(-1.3, -0.6);
  Eigen::MatrixXcd m(2, 2);
  m << c_t(1.1, 0.3), c_t(-0.5, 2.2), c_t(0.8, -1.4), c_t(-2.6, -0.9);
  std::vector<c_t> a{z, c_t(0.4, 2.9)};
  std::vector<Eigen::VectorXcd> av{v, v.reverse()};
  std::vector<Eigen::RowVectorXcd> arv{rv, rv.reverse()};
  std::vector<Eigen::MatrixXcd> am{m, m.transpose()};
  std::vector<std::vector<c_t>> aa{a, {c_t(-0.7, 1.8), z}};

  expect_rep(z);
  expect_rep(v);
  expect_rep(rv);
  expect_rep(m);
  expect_rep(a);
  expect_rep(av);
  expect_rep(arv);
  expect_rep(am);
  expect_rep(aa);
}

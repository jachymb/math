#include <test/unit/math/test_ad.hpp>
#include <cmath>
#include <limits>

TEST(mathMixMatFun, log1pExp) {
  auto f = [](const auto& x1) { return stan::math::log1p_exp(x1); };
  stan::test::expect_common_nonzero_unary_vectorized<
      stan::test::ScalarSupport::Real>(f);
  stan::test::expect_unary_vectorized(f, -2.6, -2, -1, -0.5, -0.2, 0.5, 1.0,
                                      1.3, 2, 3);

  std::vector<double> com_args = stan::test::internal::common_nonzero_args();
  std::vector<double> args{0.1, -2.5, 5.5};

  stan::test::expect_ad_vector_matvar(f, stan::math::to_vector(com_args));
  stan::test::expect_ad_vector_matvar(f, stan::math::to_vector(args));
}

TEST(mathMixScalFun, log1pExpHigherDerivativesLargeArgs) {
  using stan::math::fvar;
  using stan::math::log1p_exp;
  using stan::math::var;
  for (double x : {-800, -40, -20, -1, 0, 1, 20, 30, 37, 40, 100, 700, 800}) {
    // log1p_exp' = inv_logit; compare with closed forms, e = exp(-|x|)
    double e = std::exp(-std::fabs(x));
    double d2 = e / ((1 + e) * (1 + e));
    double d3 = (x > 0 ? 1 : -1) * d2 * std::expm1(-std::fabs(x)) / (1 + e);
    double tol2 = 1e-13 * std::fmax(d2, std::numeric_limits<double>::min());
    double tol3
        = 1e-13 * std::fmax(std::fabs(d3), std::numeric_limits<double>::min());

    fvar<fvar<double>> ff(fvar<double>(x, 1), fvar<double>(1, 0));
    EXPECT_NEAR(log1p_exp(ff).d_.d_, d2, tol2) << "x = " << x;

    fvar<var> fv(x, 1);
    log1p_exp(fv).d_.grad();
    EXPECT_NEAR(fv.val_.adj(), d2, tol2) << "x = " << x;
    stan::math::recover_memory();

    fvar<fvar<var>> ffv(fvar<var>(x, 1), fvar<var>(1, 0));
    log1p_exp(ffv).d_.d_.grad();
    EXPECT_NEAR(ffv.val_.val_.adj(), d3, tol3) << "x = " << x;
    stan::math::recover_memory();
  }
}

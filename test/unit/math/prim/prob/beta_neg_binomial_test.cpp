#include <stan/math/prim.hpp>
#include <test/unit/math/prim/prob/vector_rng_test_helper.hpp>
#include <test/unit/math/prim/prob/VectorIntRNGTestRig.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <test/unit/math/prim/prob/util.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

class BetaNegBinomialTestRig : public VectorIntRNGTestRig {
 public:
  BetaNegBinomialTestRig()
      : VectorIntRNGTestRig(10000, 10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                            {1.1, 3.1, 8.1}, {1, 3, 8}, {-3.0, -2.0, 0.0},
                            {-3, -2, 0}, {3.1, 4.1, 5.1}, {3, 4, 5},
                            {-2.1, -0.5, 0.0}, {-3, -1, 0}, {1.1, 3.1, 8.1},
                            {1, 3, 8}, {-3.0, -2.0, 0.0}, {-3, -2, 0}) {}

  template <typename T1, typename T2, typename T3, typename T_rng>
  auto generate_samples(const T1& N, const T2& alpha, const T3& beta,
                        T_rng& rng) const {
    return stan::math::beta_neg_binomial_rng(N, alpha, beta, rng);
  }

  template <typename T1>
  double pmf(int y, T1 r, double alpha, double beta) const {
    return std::exp(stan::math::beta_neg_binomial_lpmf(y, r, alpha, beta));
  }
};

TEST(ProbDistributionsBetaNegBinomial, errorCheck) {
  check_dist_throws_int_first_argument(BetaNegBinomialTestRig());
}

TEST(ProbDistributionsBetaNegBinomial, distributionCheck) {
  check_counts_int_real_real(BetaNegBinomialTestRig());
}

TEST(ProbDistributionBetaNegBinomial, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::beta_neg_binomial_rng(4, 0.6, 2.0, rng));

  EXPECT_THROW(stan::math::beta_neg_binomial_rng(-4, 0.6, 2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::beta_neg_binomial_rng(4, -0.6, 2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::beta_neg_binomial_rng(4, 0.6, -2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::beta_neg_binomial_rng(
                   4, stan::math::positive_infinity(), 2, rng),
               std::domain_error);
  EXPECT_THROW(stan::math::beta_neg_binomial_rng(
                   4, 0.6, stan::math::positive_infinity(), rng),
               std::domain_error);
}

TEST(ProbDistributionBetaNegBinomial, rngSmallBeta) {
  // About half of the beta draws round to p = 1 here, which used to make the
  // odds ratio p / (1 - p) infinite and throw
  boost::random::mt19937 rng;
  for (int i = 0; i < 100; ++i) {
    ASSERT_NO_THROW(stan::math::beta_neg_binomial_rng(1, 3, 0.02, rng));
  }
}

TEST(ProbDistributionBetaNegBinomial, rngVectorDrawsIndependent) {
  // All elements used to share one beta draw, which made their correlation
  // 0.42 for these parameters
  boost::random::mt19937 rng;
  std::vector<double> r(2, 5.0);
  std::vector<double> y0;
  std::vector<double> y1;
  for (int i = 0; i < 10000; ++i) {
    std::vector<int> y = stan::math::beta_neg_binomial_rng(r, 8.0, 8.0, rng);
    y0.push_back(y[0]);
    y1.push_back(y[1]);
  }
  EXPECT_NEAR(0, sample_correlation(y0, y1), 0.1);
}

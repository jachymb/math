#ifndef TEST_UNIT_MATH_PRIM_PROB_HPP
#define TEST_UNIT_MATH_PRIM_PROB_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <boost/math/distributions.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

/**
 * Uses a chi-squared test to assert that a vector of observed counts
 * is consistent with a vector of expected counts. Useful for testing RNGs.
 */
inline void assert_chi_squared(const std::vector<int>& counts,
                               const std::vector<double>& expected,
                               double tolerance) {
  int bins = counts.size();
  EXPECT_EQ(bins, expected.size());

  double chi = 0;
  for (int i = 0; i < bins; ++i) {
    double discrepancy = expected[i] - counts[i];
    chi += discrepancy * discrepancy / expected[i];
  }
  boost::math::chi_squared dist(bins - 1);
  double chi_threshold = quantile(complement(dist, tolerance));

  EXPECT_TRUE(chi < chi_threshold);
}

/**
 * Like assert_matches_quantiles, but the bins are not necessarily
 * equiprobable. Assert that approximately proportions[i] of the
 * samples are in bin i, which has lower bound bin_boundaries[i-1] and
 * upper bound bin_boundaries[i], using a chi-squared goodness of fit
 * test. bin_boundaries is assumed sorted in increasing order.
 **/
inline void assert_matches_bins(const std::vector<double>& samples,
                                const std::vector<double>& bin_boundaries,
                                const std::vector<double>& proportions,
                                double tolerance) {
  ASSERT_GT(samples.size(), 0);
  int N = samples.size();
  std::vector<double> mysamples = samples;
  std::sort(mysamples.begin(), mysamples.end());

  ASSERT_GT(bin_boundaries.size(), 0);
  ASSERT_TRUE(bin_boundaries.size() == proportions.size());
  int K = bin_boundaries.size();
  std::vector<double> expected;
  for (int i = 0; i < K; i++) {
    ASSERT_TRUE(proportions[i] >= 0 && proportions[i] <= 1);
    expected.push_back(proportions[i] * N);
  }

  std::vector<int> counts(K);
  size_t current_index = 0;
  for (int i = 0; i < N; ++i) {
    while (mysamples[i] >= bin_boundaries[current_index]) {
      ++current_index;
      EXPECT_TRUE(current_index < bin_boundaries.size());
    }
    ++counts[current_index];
  }
  assert_chi_squared(counts, expected, tolerance);
}

/**
 * From a collection of samples and a list of quantiles, assumed
 * ordered, assert that the samples resemble draws from a distribution
 * with those quantiles, using a chi_squared goodness of fit
 * test. That is, assert that the samples are approximately evenly
 * distributed among the quantiles.size() equiprobable bins, who's
 * upper bounds are given in quantiles in increasing order.
 */
inline void assert_matches_quantiles(const std::vector<double>& samples,
                                     const std::vector<double>& quantiles,
                                     double tolerance) {
  int K = quantiles.size();
  std::vector<double> proportions;
  for (int i = 0; i < K; ++i)
    proportions.push_back(1.0 / K);

  assert_matches_bins(samples, quantiles, proportions, tolerance);
}

/**
 * Return the sample correlation of two equally long sequences. Useful for
 * testing that the elements of a vectorized RNG call are independent.
 */
inline double sample_correlation(const std::vector<double>& x,
                                 const std::vector<double>& y) {
  Eigen::Map<const Eigen::ArrayXd> x_arr(x.data(), x.size());
  Eigen::Map<const Eigen::ArrayXd> y_arr(y.data(), y.size());
  Eigen::ArrayXd x_c = x_arr - x_arr.mean();
  Eigen::ArrayXd y_c = y_arr - y_arr.mean();
  return (x_c * y_c).sum() / std::sqrt(x_c.square().sum() * y_c.square().sum());
}
#endif

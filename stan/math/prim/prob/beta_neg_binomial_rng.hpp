#ifndef STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_RNG_HPP
#define STAN_MATH_PRIM_PROB_BETA_NEG_BINOMIAL_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/prob/gamma_rng.hpp>
#include <stan/math/prim/prob/neg_binomial_rng.hpp>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return a beta-negative binomial random variate with the given number of
 * successes, prior success, and prior failure parameters, using the given
 * random number generator.
 *
 * r, alpha, and beta can each be a scalar or a one-dimensional container. Any
 * non-scalar inputs must be the same size.
 *
 * @tparam T_r type of number of successes parameter
 * @tparam T_alpha type of prior success parameter
 * @tparam T_beta type of prior failure parameter
 * @tparam RNG type of random number generator
 *
 * @param r (Sequence of) number of successes parameter(s)
 * @param alpha (Sequence of) prior success parameter(s)
 * @param beta (Sequence of) prior failure parameter(s)
 * @param rng random number generator
 * @return (Sequence of) beta-negative binomial random variate(s)
 * @throw std::domain_error if r, alpha, or beta are nonpositive, or if a
 * Poisson rate drawn from the gamma distribution is not less than
 * `POISSON_MAX_RATE` (this is common for small alpha, where the distribution
 * has a heavy tail)
 * @throw std::invalid_argument if non-scalar arguments are of different sizes
 */
template <typename T_r, typename T_alpha, typename T_beta, typename RNG>
inline typename VectorBuilder<true, int, T_r, T_alpha, T_beta>::type
beta_neg_binomial_rng(const T_r &r, const T_alpha &alpha, const T_beta &beta,
                      RNG &rng) {
  using T_r_ref = ref_type_t<T_r>;
  using T_alpha_ref = ref_type_t<T_alpha>;
  using T_beta_ref = ref_type_t<T_beta>;
  static constexpr const char *function = "beta_neg_binomial_rng";
  check_consistent_sizes(function, "Number of successes parameter", r,
                         "Prior success parameter", alpha,
                         "Prior failure parameter", beta);

  T_r_ref r_ref = r;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;
  check_positive_finite(function, "Number of successes parameter", r_ref);
  check_positive_finite(function, "Prior success parameter", alpha_ref);
  check_positive_finite(function, "Prior failure parameter", beta_ref);
  if (size_zero(r, alpha, beta)) {
    return {};
  }

  scalar_seq_view<T_r_ref> r_vec(r_ref);
  scalar_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_beta_ref> beta_vec(beta_ref);
  size_t N = max_size(r, alpha, beta);
  VectorBuilder<true, int, T_r, T_alpha, T_beta> output(N);

  for (size_t n = 0; n < N; ++n) {
    // For p ~ beta(alpha, beta), (1 - p) / p is the ratio of independent
    // gamma draws with shapes beta and alpha. Drawing them directly avoids
    // rounding p to 1, where the odds ratio p / (1 - p) overflows.
    double g_beta = internal::gamma_draw(beta_vec[n], rng);
    double g_alpha = internal::gamma_draw(alpha_vec[n], rng);
    output[n] = internal::gamma_poisson_draw(function, r_vec[n],
                                             g_beta / g_alpha, rng);
  }

  return output.data();
}

}  // namespace math
}  // namespace stan
#endif

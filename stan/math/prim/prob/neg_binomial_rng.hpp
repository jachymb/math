#ifndef STAN_MATH_PRIM_PROB_NEG_BINOMIAL_RNG_HPP
#define STAN_MATH_PRIM_PROB_NEG_BINOMIAL_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/prob/gamma_rng.hpp>
#include <stan/math/prim/prob/poisson_rng.hpp>

namespace stan {
namespace math {

namespace internal {

/**
 * Return a negative binomial random variate as a Poisson draw whose rate is
 * drawn from the gamma distribution with the specified shape and scale.
 *
 * @tparam RNG type of random number generator
 * @param function name of the calling function, for error messages
 * @param shape positive finite shape of the gamma distribution
 * @param scale scale of the gamma distribution
 * @param rng random number generator
 * @return negative binomial random variate
 * @throw std::domain_error if the drawn rate is not less than
 * `POISSON_MAX_RATE`, so that the draw may not fit in an `int`, or is NaN
 */
template <typename RNG>
inline int gamma_poisson_draw(const char* function, double shape, double scale,
                              RNG& rng) {
  double lambda = gamma_draw(shape, rng) * scale;
  check_less(function, "Poisson rate drawn from the gamma distribution", lambda,
             POISSON_MAX_RATE);
  return poisson_draw(lambda, rng);
}

}  // namespace internal

/** \ingroup prob_dists
 * Return a negative binomial random variate with the specified shape and
 * inverse scale parameters using the given random number generator.
 *
 * alpha and beta can each be a scalar or a one-dimensional container. Any
 * non-scalar inputs must be the same size.
 *
 * @tparam T_shape type of shape parameter
 * @tparam T_inv type of inverse scale parameter
 * @tparam RNG type of random number generator
 * @param alpha (Sequence of) positive shape parameter(s)
 * @param beta (Sequence of) positive inverse scale parameter(s)
 * @param rng random number generator
 * @return (Sequence of) negative binomial random variate(s)
 * @throw std::domain_error if alpha or beta are nonpositive, or if a Poisson
 * rate drawn from the gamma distribution is not less than `POISSON_MAX_RATE`
 * @throw std::invalid_argument if non-scalar arguments are of different
 * sizes
 */
template <typename T_shape, typename T_inv, class RNG>
inline typename VectorBuilder<true, int, T_shape, T_inv>::type neg_binomial_rng(
    const T_shape& alpha, const T_inv& beta, RNG& rng) {
  using T_alpha_ref = ref_type_t<T_shape>;
  using T_beta_ref = ref_type_t<T_inv>;
  static constexpr const char* function = "neg_binomial_rng";
  check_consistent_sizes(function, "Shape parameter", alpha,
                         "Inverse scale Parameter", beta);
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;

  if (size_zero(alpha, beta)) {
    return {};
  }

  check_positive_finite(function, "Shape parameter", alpha_ref);
  check_positive_finite(function, "Inverse scale parameter", beta_ref);

  scalar_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_beta_ref> beta_vec(beta_ref);
  size_t N = max_size(alpha, beta);
  VectorBuilder<true, int, T_shape, T_inv> output(N);

  for (size_t n = 0; n < N; ++n) {
    output[n] = internal::gamma_poisson_draw(function, alpha_vec[n],
                                             1.0 / beta_vec[n], rng);
  }

  return output.data();
}

}  // namespace math
}  // namespace stan
#endif

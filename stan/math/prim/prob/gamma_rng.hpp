#ifndef STAN_MATH_PRIM_PROB_GAMMA_RNG_HPP
#define STAN_MATH_PRIM_PROB_GAMMA_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/boost_policy.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <cmath>

namespace stan {
namespace math {

namespace internal {

/**
 * Return a gamma random variate with the specified shape and unit scale.
 *
 * Below a shape of \f$ 2^{32} \f$ this is Boost's sampler. Above it, the
 * exponent of Boost's acceptance test loses about \f$ 10^{-16} \alpha \f$ to
 * cancellation, which spreads the draws 2.8 times too wide at
 * \f$ \alpha = 10^{18} \f$. There the method of Marsaglia and Tsang (2000) is
 * used instead, with its log acceptance ratio
 * \f$ x^2 / 2 + d (1 - v + \log v) \f$, \f$ v = (1 + t)^3 \f$, written as
 * \f$ x^2 / 2 + d (3\,\mathrm{log1pmx}(t) - t^2 (3 + t)) \f$ so that it
 * stays accurate for any shape.
 *
 * @tparam RNG type of random number generator
 * @param alpha positive finite shape
 * @param rng random number generator
 * @return gamma random variate
 */
template <typename RNG>
inline double gamma_draw(double alpha, RNG& rng) {
  if (alpha < 4294967296.0) {
    return boost::random::gamma_distribution<>(alpha)(rng);
  }
  const double d = alpha - 1.0 / 3.0;
  const double c = 1.0 / (3.0 * std::sqrt(d));
  boost::random::normal_distribution<> std_normal;
  boost::random::uniform_01<> uniform;
  while (true) {
    const double x = std_normal(rng);
    const double t = c * x;
    const double log1pmx_t = boost::math::log1pmx(t, boost_policy_t<>());
    const double log_accept
        = 0.5 * x * x + d * (3.0 * log1pmx_t - t * t * (3.0 + t));
    if (std::log(uniform(rng)) < log_accept) {
      return d + d * t * (3.0 + t * (3.0 + t));
    }
  }
}

}  // namespace internal

/** \ingroup prob_dists
 * Return a gamma random variate for the given shape and inverse
 * scale parameters using the specified random number generator.
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
 * @return (Sequence of) gamma random variate(s)
 * @throw std::domain_error if alpha or beta are nonpositive
 * @throw std::invalid_argument if non-scalar arguments are of different
 * sizes
 */
template <typename T_shape, typename T_inv, class RNG>
inline typename VectorBuilder<true, double, T_shape, T_inv>::type gamma_rng(
    const T_shape& alpha, const T_inv& beta, RNG& rng) {
  using T_alpha_ref = ref_type_t<T_shape>;
  using T_beta_ref = ref_type_t<T_inv>;
  static constexpr const char* function = "gamma_rng";
  check_consistent_sizes(function, "Shape parameter", alpha,
                         "Inverse scale Parameter", beta);
  if (size_zero(alpha, beta)) {
    return {};
  }

  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;
  check_positive_finite(function, "Shape parameter", alpha_ref);
  check_positive_finite(function, "Inverse scale parameter", beta_ref);

  scalar_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_beta_ref> beta_vec(beta_ref);
  size_t N = max_size(alpha, beta);
  VectorBuilder<true, double, T_shape, T_inv> output(N);

  for (size_t n = 0; n < N; ++n) {
    // Convert rate (inverse scale) argument to scale
    output[n] = internal::gamma_draw(alpha_vec[n], rng)
                * (1 / static_cast<double>(beta_vec[n]));
  }

  return output.data();
}

}  // namespace math
}  // namespace stan
#endif

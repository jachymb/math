#ifndef STAN_MATH_PRIM_PROB_YULE_SIMON_RNG_HPP
#define STAN_MATH_PRIM_PROB_YULE_SIMON_RNG_HPP

#include <utility>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/prob/exponential_rng.hpp>
#include <stan/math/prim/prob/neg_binomial_rng.hpp>
#include <cmath>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return a yule-simon random variate with the given shape parameter,
 * using the given random number generator.
 *
 * alpha can be a scalar or a one-dimensional container.
 *
 * @tparam T_alpha type of shape parameter
 * @tparam RNG type of random number generator
 *
 * @param alpha (Sequence of) shape parameter(s)
 * @param rng random number generator
 * @return (Sequence of) yule-simon random variate(s)
 * @throw std::domain_error if alpha is nonpositive, or if a Poisson rate
 * drawn from the gamma distribution is not less than `POISSON_MAX_RATE`
 */
template <typename T_alpha, typename RNG>
inline auto yule_simon_rng(T_alpha&& alpha, RNG& rng) {
  static constexpr const char* function = "yule_simon_rng";
  decltype(auto) alpha_ref = to_ref(std::forward<T_alpha>(alpha));
  check_positive_finite(function, "Shape parameter", alpha_ref);

  auto w = exponential_rng(std::forward<decltype(alpha_ref)>(alpha_ref), rng);
  scalar_seq_view<decltype(w)> w_vec(w);
  size_t N = stan::math::size(w);
  VectorBuilder<true, int, decltype(w)> output(N);
  for (size_t n = 0; n < N; ++n) {
    // The draw minus 1 is geometric with success probability p = exp(-w), a
    // negative binomial whose inverse odds ratio (1 - p) / p is expm1(w);
    // computing it from p would round p to 1 for tiny w
    output[n]
        = internal::gamma_poisson_draw(function, 1.0, std::expm1(w_vec[n]), rng)
          + 1;
  }
  return output.data();
}

}  // namespace math
}  // namespace stan
#endif

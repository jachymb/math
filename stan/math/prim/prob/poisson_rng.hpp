#ifndef STAN_MATH_PRIM_PROB_POISSON_RNG_HPP
#define STAN_MATH_PRIM_PROB_POISSON_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <boost/random/poisson_distribution.hpp>

namespace stan {
namespace math {

namespace internal {

/**
 * Return a Poisson random variate with the specified rate, which the caller
 * has checked to be nonnegative and less than `POISSON_MAX_RATE`.
 *
 * @tparam RNG type of random number generator
 * @param lambda rate
 * @param rng random number generator
 * @return Poisson random variate
 */
template <typename RNG>
inline int poisson_draw(double lambda, RNG& rng) {
  // Boost requires a positive rate
  if (lambda == 0) {
    return 0;
  }
  return boost::random::poisson_distribution<>(lambda)(rng);
}

}  // namespace internal

/** \ingroup prob_dists
 * Return a Poisson random variate with specified rate parameter
 * using the given random number generator.
 *
 * lambda can be a scalar or a one-dimensional container.
 *
 * @tparam T_rate type of rate parameter
 * @tparam RNG type of random number generator
 * @param lambda (Sequence of) rate parameter(s)
 * @param rng random number generator
 * @return (Sequence of) Poisson random variate(s)
 * @throw std::domain_error if lambda is negative or not less than
 * `POISSON_MAX_RATE`, \f$ 2^{31} - 2^{21} \f$
 */
template <typename T_rate, class RNG>
inline typename VectorBuilder<true, int, T_rate>::type poisson_rng(
    const T_rate& lambda, RNG& rng) {
  static constexpr const char* function = "poisson_rng";
  const auto& lambda_ref = to_ref(lambda);
  check_nonnegative(function, "Rate parameter", lambda_ref);
  check_less(function, "Rate parameter", lambda_ref, POISSON_MAX_RATE);

  scalar_seq_view<T_rate> lambda_vec(lambda_ref);
  size_t N = stan::math::size(lambda);
  VectorBuilder<true, int, T_rate> output(N);

  for (size_t n = 0; n < N; ++n) {
    output[n] = internal::poisson_draw(lambda_vec[n], rng);
  }

  return output.data();
}

}  // namespace math
}  // namespace stan
#endif

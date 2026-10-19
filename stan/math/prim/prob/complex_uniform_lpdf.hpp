#ifndef STAN_MATH_PRIM_PROB_COMPLEX_UNIFORM_LPDF_HPP
#define STAN_MATH_PRIM_PROB_COMPLEX_UNIFORM_LPDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/get_imag.hpp>
#include <stan/math/prim/fun/get_real.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <complex>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return the log of the complex uniform density of `y`, which is uniform on
 * the closed disk of radius `rho` around `mu` in the complex plane:
 *
 * \f[
 * \log p(y \mid \mu, \rho) = -\log \pi - 2 \log \rho
 * \quad \textrm{if } |y - \mu| \le \rho,
 * \f]
 *
 * and \f$-\infty\f$ otherwise. The density is with respect to the Lebesgue
 * measure on \f$(\textrm{Re}\, y, \textrm{Im}\, y)\f$.
 *
 * @tparam T_y type of the variate, a complex scalar
 * @tparam T_loc type of the centre, a complex scalar
 * @tparam T_scale type of the radius, a real scalar
 * @param y complex variate
 * @param mu centre of the disk
 * @param rho radius of the disk
 * @return log density
 * @throw std::domain_error if a part of `y` is NaN, a part of `mu` is not
 * finite, or `rho` is not positive and finite
 */
template <bool propto, typename T_y, typename T_loc, typename T_scale,
          require_all_complex_t<T_y, T_loc>* = nullptr,
          require_stan_scalar_t<T_scale>* = nullptr,
          require_not_complex_t<T_scale>* = nullptr>
inline real_return_t<T_y, T_loc, T_scale> complex_uniform_lpdf(T_y&& y,
                                                               T_loc&& mu,
                                                               T_scale&& rho) {
  // the density depends on y and mu only through the support
  using T_partials_return = partials_return_t<T_scale>;
  static constexpr const char* function = "complex_uniform_lpdf";
  check_not_nan(function, "Real part of random variable", get_real(y));
  check_not_nan(function, "Imaginary part of random variable", get_imag(y));
  check_finite(function, "Real part of location parameter", get_real(mu));
  check_finite(function, "Imaginary part of location parameter", get_imag(mu));
  check_positive_finite(function, "Radius parameter", rho);

  if constexpr (!include_summand<propto, base_type_t<T_y>, base_type_t<T_loc>,
                                 T_scale>::value) {
    return 0.0;
  }
  if (std::abs(value_of_rec(y) - value_of_rec(mu)) > value_of_rec(rho)) {
    return LOG_ZERO;
  }

  T_partials_return logp = 0;
  if constexpr (include_summand<propto>::value) {
    logp -= LOG_PI;
  }
  if constexpr (include_summand<propto, T_scale>::value) {
    logp -= 2.0 * log(value_of(rho));
  }
  auto ops_partials = make_partials_propagator(rho);
  if constexpr (is_autodiff_v<T_scale>) {
    partials<0>(ops_partials) = -2.0 / value_of(rho);
  }
  return ops_partials.build(logp);
}

template <typename T_y, typename T_loc, typename T_scale>
inline real_return_t<T_y, T_loc, T_scale> complex_uniform_lpdf(T_y&& y,
                                                               T_loc&& mu,
                                                               T_scale&& rho) {
  return complex_uniform_lpdf<false>(std::forward<T_y>(y),
                                     std::forward<T_loc>(mu),
                                     std::forward<T_scale>(rho));
}

}  // namespace math
}  // namespace stan
#endif

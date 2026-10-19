#ifndef STAN_MATH_PRIM_PROB_COMPLEX_UNIFORM_RNG_HPP
#define STAN_MATH_PRIM_PROB_COMPLEX_UNIFORM_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>
#include <complex>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return a complex uniform random variate, a point drawn uniformly from the
 * disk of radius `rho` around `mu`, using the specified random number
 * generator. The point is \f$\mu + \rho \sqrt{U_1} e^{2 \pi i U_2}\f$ for
 * independent standard uniform \f$U_1\f$ and \f$U_2\f$.
 *
 * @tparam T_loc type of the centre, a complex scalar
 * @tparam T_scale type of the radius, a real scalar
 * @tparam RNG type of random number generator
 * @param mu centre of the disk
 * @param rho radius of the disk
 * @param rng random number generator
 * @return complex uniform random variate
 * @throw std::domain_error if a part of `mu` is not finite or `rho` is not
 * positive and finite
 */
template <typename T_loc, typename T_scale, typename RNG,
          require_complex_bt<std::is_arithmetic, T_loc>* = nullptr,
          require_arithmetic_t<T_scale>* = nullptr>
inline std::complex<double> complex_uniform_rng(T_loc&& mu, T_scale&& rho,
                                                RNG& rng) {
  using boost::variate_generator;
  using boost::random::uniform_01;
  static constexpr const char* function = "complex_uniform_rng";
  check_finite(function, "Real part of location parameter", mu.real());
  check_finite(function, "Imaginary part of location parameter", mu.imag());
  check_positive_finite(function, "Radius parameter", rho);

  variate_generator<RNG&, uniform_01<>> uniform01_rng(rng, uniform_01<>());
  const double radius = rho * std::sqrt(uniform01_rng());
  return std::complex<double>(mu)
         + std::polar(radius, TWO_PI * uniform01_rng());
}

}  // namespace math
}  // namespace stan
#endif

#ifndef STAN_MATH_PRIM_PROB_STUDENT_T_RNG_HPP
#define STAN_MATH_PRIM_PROB_STUDENT_T_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/prob/gamma_rng.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cmath>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return a student-t random variate for the given degrees of freedom,
 * location, and scale using the specified random number generator.
 *
 * nu, mu, and sigma can each be a scalar or a one-dimensional container. Any
 * non-scalar inputs must be the same size.
 *
 * @tparam T_deg type of degrees of freedom parameter
 * @tparam T_loc type of location parameter
 * @tparam T_scale type of scale parameter
 * @tparam RNG type of random number generator
 *
 * @param nu (Sequence of) degrees of freedom parameter(s)
 * @param mu (Sequence of) location parameter(s)
 * @param sigma (Sequence of) scale parameter(s)
 * @param rng random number generator
 * @return Student-t random variate
 * @throw std::domain_error if nu is nonpositive, mu is infinite, or sigma
 * is nonpositive
 * @throw std::invalid_argument if non-scalar arguments are of different
 * sizes
 */
template <typename T_deg, typename T_loc, typename T_scale, class RNG>
inline typename VectorBuilder<true, double, T_deg, T_loc, T_scale>::type
student_t_rng(const T_deg& nu, const T_loc& mu, const T_scale& sigma,
              RNG& rng) {
  using T_nu_ref = ref_type_t<T_deg>;
  using T_mu_ref = ref_type_t<T_loc>;
  using T_sigma_ref = ref_type_t<T_scale>;
  static constexpr const char* function = "student_t_rng";
  check_consistent_sizes(function, "Degrees of freedom parameter", nu,
                         "Location parameter", mu, "Scale Parameter", sigma);
  if (size_zero(nu, mu, sigma)) {
    return {};
  }
  T_nu_ref nu_ref = nu;
  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  check_positive_finite(function, "Degrees of freedom parameter", nu_ref);
  check_finite(function, "Location parameter", mu_ref);
  check_positive_finite(function, "Scale parameter", sigma_ref);

  scalar_seq_view<T_nu_ref> nu_vec(nu_ref);
  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  size_t N = max_size(nu, mu, sigma);
  VectorBuilder<true, double, T_deg, T_loc, T_scale> output(N);

  boost::random::normal_distribution<> std_normal;
  for (size_t n = 0; n < N; ++n) {
    // Boost's student_t_distribution, with the chi-square draw routed
    // through internal::gamma_draw
    double z = std_normal(rng);
    double chi_square = 2 * internal::gamma_draw(nu_vec[n] / 2.0, rng);
    output[n]
        = mu_vec[n] + sigma_vec[n] * (z / std::sqrt(chi_square / nu_vec[n]));
  }

  return output.data();
}

}  // namespace math
}  // namespace stan
#endif

#ifndef STAN_MATH_PRIM_PROB_COMPLEX_UNIFORM_LCDF_HPP
#define STAN_MATH_PRIM_PROB_COMPLEX_UNIFORM_LCDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/asin.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/get_imag.hpp>
#include <stan/math/prim/fun/get_real.hpp>
#include <stan/math/prim/fun/inv.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/square.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * Return the log of the complex uniform cumulative distribution function,
 * the joint cdf of the real and imaginary parts,
 *
 * \f[
 * F(y) = P(\textrm{Re}\, Z \le \textrm{Re}\, y,\ \textrm{Im}\, Z \le
 * \textrm{Im}\, y),
 * \f]
 *
 * for \f$Z\f$ uniform on the disk of radius `rho` around `mu`. With the
 * standardized corner \f$u + i v = (y - \mu) / \rho\f$ and
 * \f$G(s) = s \sqrt{1 - s^2} + \arcsin s + \pi / 2\f$, the area of the unit
 * disk left of \f$s\f$, the area of the unit disk in the quadrant is
 * \f$(G(u) + G(v)) / 2 - \pi / 4 + u v\f$ for a corner inside the disk and
 * \f$G(u) + G(v) - \pi\f$ for one beyond the arc with \f$u, v > 0\f$;
 * \f$F\f$ is that area over \f$\pi\f$. When only one of the two cuts bounds
 * the region, \f$F\f$ is \f$G(u) / \pi\f$ or \f$G(v) / \pi\f$.
 *
 * \f$F\f$ has an absolute error of about \f$10^{-16}\f$, so the result loses
 * relative accuracy where \f$F\f$ is tiny, at corners near the left or lower
 * edge of the disk, and can be \f$-\infty\f$ where \f$F\f$ is below about
 * \f$10^{-16}\f$.
 *
 * @tparam T_y type of the variate, a complex scalar
 * @tparam T_loc type of the centre, a complex scalar
 * @tparam T_scale type of the radius, a real scalar
 * @param y complex variate
 * @param mu centre of the disk
 * @param rho radius of the disk
 * @return log of the cdf
 * @throw std::domain_error if a part of `y` is NaN, a part of `mu` is not
 * finite, or `rho` is not positive and finite
 */
template <typename T_y, typename T_loc, typename T_scale,
          require_all_complex_t<T_y, T_loc>* = nullptr,
          require_stan_scalar_t<T_scale>* = nullptr,
          require_not_complex_t<T_scale>* = nullptr>
inline real_return_t<T_y, T_loc, T_scale> complex_uniform_lcdf(T_y&& y,
                                                               T_loc&& mu,
                                                               T_scale&& rho) {
  using T_y_real = base_type_t<T_y>;
  using T_mu_real = base_type_t<T_loc>;
  using T_partials_return = partials_return_t<T_y_real, T_mu_real, T_scale>;
  static constexpr const char* function = "complex_uniform_lcdf";
  const auto y_re = get_real(y);
  const auto y_im = get_imag(y);
  const auto mu_re = get_real(mu);
  const auto mu_im = get_imag(mu);
  check_not_nan(function, "Real part of random variable", y_re);
  check_not_nan(function, "Imaginary part of random variable", y_im);
  check_finite(function, "Real part of location parameter", mu_re);
  check_finite(function, "Imaginary part of location parameter", mu_im);
  check_positive_finite(function, "Radius parameter", rho);

  const T_partials_return rho_val = value_of(rho);
  const T_partials_return u = (value_of(y_re) - value_of(mu_re)) / rho_val;
  const T_partials_return v = (value_of(y_im) - value_of(mu_im)) / rho_val;
  const double r_sq = square(value_of_rec(u)) + square(value_of_rec(v));
  if (u <= -1 || v <= -1 || (u <= 0 && v <= 0 && r_sq >= 1)) {
    return NEGATIVE_INFTY;
  }
  if (u >= 1 && v >= 1) {
    return 0.0;
  }

  // half chord of the unit circle at s, and area of the unit disk left of s,
  // for |s| < 1
  const auto half_chord = [](const auto& s) { return sqrt((1 - s) * (1 + s)); };
  const auto left_area = [&half_chord](const auto& s) {
    return s * half_chord(s) + asin(s) + 0.5 * pi();
  };
  // area of the unit disk in the quadrant, its derivatives in u and v, and
  // rho times its derivative in rho; a cut that does not bind (u or v may be
  // infinite there) enters none of them
  T_partials_return area;
  T_partials_return d_u = 0;
  T_partials_return d_v = 0;
  T_partials_return d_rho;
  if (v >= 1 || (u <= 0 && r_sq >= 1)) {
    // the cut at v misses the part of the disk left of u
    area = left_area(u);
    d_u = 2 * half_chord(u);
    d_rho = -u * d_u;
  } else if (u >= 1 || (v <= 0 && r_sq >= 1)) {
    // the cut at u misses the part of the disk below v
    area = left_area(v);
    d_v = 2 * half_chord(v);
    d_rho = -v * d_v;
  } else {
    if (r_sq <= 1) {
      area = 0.5 * (left_area(u) + left_area(v)) - 0.25 * pi() + u * v;
      d_u = half_chord(u) + v;
      d_v = half_chord(v) + u;
    } else {
      // u, v > 0 beyond the arc: the two caps cut off are disjoint
      area = left_area(u) + left_area(v) - pi();
      d_u = 2 * half_chord(u);
      d_v = 2 * half_chord(v);
    }
    d_rho = -(u * d_u + v * d_v);
  }
  if (area <= 0) {
    // rounding where the true area is of the order of 1e-16 or less
    return NEGATIVE_INFTY;
  }

  auto ops_partials = make_partials_propagator(y_re, y_im, mu_re, mu_im, rho);
  if constexpr (is_any_autodiff_v<T_y_real, T_mu_real, T_scale>) {
    // derivatives of log F in Re y and Im y
    const T_partials_return inv_rho_area = inv(rho_val * area);
    const T_partials_return d_re = d_u * inv_rho_area;
    const T_partials_return d_im = d_v * inv_rho_area;
    if constexpr (is_autodiff_v<T_y_real>) {
      partials<0>(ops_partials) = d_re;
      partials<1>(ops_partials) = d_im;
    }
    if constexpr (is_autodiff_v<T_mu_real>) {
      partials<2>(ops_partials) = -d_re;
      partials<3>(ops_partials) = -d_im;
    }
    if constexpr (is_autodiff_v<T_scale>) {
      partials<4>(ops_partials) = d_rho * inv_rho_area;
    }
  }
  return ops_partials.build(log(area) - LOG_PI);
}

}  // namespace math
}  // namespace stan
#endif

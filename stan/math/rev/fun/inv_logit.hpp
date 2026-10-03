#ifndef STAN_MATH_REV_FUN_INV_LOGIT_HPP
#define STAN_MATH_REV_FUN_INV_LOGIT_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/inv_logit.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * The inverse logit function for variables (stan).
 *
 * See inv_logit() for the double-based version.
 *
 * The derivative of inverse logit is
 *
 * \f$\frac{d}{dx} \mbox{logit}^{-1}(x) = \frac{e}{(1 + e)^2}\f$
 * with \f$e = \exp(-|x|)\f$.
 *
 * @param a Argument variable.
 * @return Inverse logit of argument.
 */
inline var inv_logit(const var& a) {
  const double e = std::exp(-std::fabs(a.val()));
  return make_callback_var((a.val() >= 0 ? 1.0 : e) / (1.0 + e),
                           [a, e](auto& vi) mutable {
                             a.adj() += vi.adj() * e / ((1.0 + e) * (1.0 + e));
                           });
}

/**
 * The inverse logit function for a `var_value` matrix.
 *
 * See inv_logit() for the double-based version.
 *
 * @tparam T type of `var_value` matrix
 * @param x argument
 * @return Inverse logit of argument.
 */
template <typename T, require_var_matrix_t<T>* = nullptr>
inline auto inv_logit(const T& x) {
  arena_t<plain_type_t<decltype(x.val().array())>> e
      = (-x.val().array().abs()).exp();
  return make_callback_var(
      ((x.val().array() >= 0).select(1.0, e) / (1.0 + e)).matrix(),
      [x, e](auto& vi) mutable {
        x.adj().array() += vi.adj().array() * e / (1.0 + e).square();
      });
}

/**
 * The inverse logit function for Eigen expressions with var value type.
 *
 * See inv_logit() for the double-based version.
 *
 * @tparam T type of Eigen expression
 * @param x Eigen expression
 * @return Inverse logit of argument.
 */
template <typename T, require_eigen_vt<is_var, T>* = nullptr>
inline auto inv_logit(T&& x) {
  auto x_arena = to_arena(std::forward<T>(x));
  using array_t = plain_type_t<decltype(x_arena.val().array())>;
  const arena_t<array_t> x_val = x_arena.val().array();
  arena_t<array_t> e = (-x_val.abs()).exp();
  arena_t<T> ret = ((x_val >= 0).select(1.0, e) / (1.0 + e)).matrix();
  reverse_pass_callback([x_arena, ret, e]() mutable {
    x_arena.adj().array() += ret.adj().array() * e / (1.0 + e).square();
  });
  return ret;
}

}  // namespace math
}  // namespace stan
#endif

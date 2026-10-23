#ifndef STAN_MATH_PRIM_FUN_LMULTIPLY_HPP
#define STAN_MATH_PRIM_FUN_LMULTIPLY_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/functor/apply_scalar_binary.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Return the first argument times the log of the second argument. The
 * result is 0 if both arguments are 0.  The function is defined by
 * `lmultiply(x, y) = x * log(y)` if `x` or `y` is non-zero and
 * `lmultiply(0, 0) = 0` otherwise.
 *
 * The autodiff overloads take the partial with respect to `y` as `x / y`,
 * or 0 when `x` is 0: `lmultiply(0, y)` is identically 0, and 0 is the
 * limit of `x / y` as `x` goes to 0 at any fixed `y > 0`. For `x != 0`
 * and `y = 0` the partial stays `x / 0`, infinite with the sign of `x`.
 * At `x = y = 0` the reverse-mode partials are therefore `log(0) = -inf`
 * and 0, and the derivative of `lmultiply(x, x)` is `-inf`, the limit of
 * `log(x) + 1` as `x` goes to 0 from above, rather than NaN. In reverse
 * mode a result with adjoint 0 likewise contributes nothing to either
 * argument, rather than `0 * log(0)` or `0 * x / 0`; this makes the second
 * derivative of `lmultiply(x, x)` at 0 through `fvar<var>` `+inf`, the
 * limit of `1 / x`. (An `fvar` first argument with zero tangent still gives
 * `0 * log(0) = NaN` there, so the second derivative with respect to `y`
 * twice at `x = y = 0` stays NaN.)
 *
 * @tparam T1 type of the first argument
 * @tparam T2 type of the second argument
 * @param a first argument
 * @param b second argument
 * @return the first argument times the log of the second argument
 */
template <typename T1, typename T2, require_all_arithmetic_t<T1, T2>* = nullptr>
inline return_type_t<T1, T2> lmultiply(const T1 a, const T2 b) {
  using std::log;
  if (a == 0 && b == 0) {
    return 0;
  }
  return a * log(b);
}

/**
 * Return the result of applying `lmultiply` to the arguments
 * elementwise, with broadcasting if one of the arguments is a scalar.
 * At least one of the arguments must be a container.
 *
 * @tparam T1 type of the first argument
 * @tparam T2 type of the second argument
 * @param a first argument
 * @param b second argument
 * @return result of applying `lmultiply` to the arguments
 */
template <typename T1, typename T2, require_any_container_t<T1, T2>* = nullptr,
          require_all_not_var_matrix_t<T1, T2>* = nullptr>
inline auto lmultiply(T1&& a, T2&& b) {
  return apply_scalar_binary(
      [](auto&& c, auto&& d) {
        return lmultiply(std::forward<decltype(c)>(c),
                         std::forward<decltype(d)>(d));
      },
      std::forward<T1>(a), std::forward<T2>(b));
}

}  // namespace math
}  // namespace stan

#endif

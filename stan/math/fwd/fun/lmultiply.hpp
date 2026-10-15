#ifndef STAN_MATH_FWD_FUN_LMULTIPLY_HPP
#define STAN_MATH_FWD_FUN_LMULTIPLY_HPP

#include <stan/math/fwd/meta.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/fun/log.hpp>
#include <stan/math/prim/fun/lmultiply.hpp>
#include <cmath>

namespace stan {
namespace math {

// The term x1 * x2.d_ / x2.val_ of the tangent is dropped only when it is
// 0/0: it is 0 whenever x1 is 0, but for a nested T it still carries the
// derivative x2.d_ / x2.val_ with respect to x1, which must be kept. A single
// return with a conditional term, rather than an early return, lets the
// compiler share the log(x2) of the value and of the tangent for T = double.
template <typename T>
inline fvar<T> lmultiply(const fvar<T>& x1, const fvar<T>& x2) {
  const bool both_zero = x1.val_ == 0.0 && x2.val_ == 0.0;
  return fvar<T>(
      lmultiply(x1.val_, x2.val_),
      x1.d_ * log(x2.val_) + (both_zero ? T(0.0) : x1.val_ * x2.d_ / x2.val_));
}

template <typename T>
inline fvar<T> lmultiply(double x1, const fvar<T>& x2) {
  const bool both_zero = x1 == 0.0 && x2.val_ == 0.0;
  return fvar<T>(lmultiply(x1, x2.val_),
                 both_zero ? T(0.0) : x1 * x2.d_ / x2.val_);
}

template <typename T>
inline fvar<T> lmultiply(const fvar<T>& x1, double x2) {
  return fvar<T>(lmultiply(x1.val_, x2), x1.d_ * log(x2));
}
}  // namespace math
}  // namespace stan
#endif

#include <stan/math/fwd.hpp>
#include <gtest/gtest.h>
#include <limits>

// The tangent of x * log(x) at x = 0 is log(0) + 0 = -inf, its limit from
// above; the partial with respect to the second argument is 0 at a = 0.
TEST(AgradFwdLmultiply, derivative_at_zero) {
  using stan::math::fvar;
  using stan::math::lmultiply;
  constexpr double inf = std::numeric_limits<double>::infinity();
  fvar<double> x(0.0, 1.0);
  EXPECT_EQ(0, lmultiply(x, x).val_);
  EXPECT_EQ(-inf, lmultiply(x, x).d_);
  EXPECT_EQ(-inf, lmultiply(x, fvar<double>(0.0)).d_);
  EXPECT_EQ(0, lmultiply(0.0, x).d_);
}

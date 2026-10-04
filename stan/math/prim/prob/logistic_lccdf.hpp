#ifndef STAN_MATH_PRIM_PROB_LOGISTIC_LCCDF_HPP
#define STAN_MATH_PRIM_PROB_LOGISTIC_LCCDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/abs.hpp>
#include <stan/math/prim/fun/as_value_column_array_or_scalar.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/inv.hpp>
#include <stan/math/prim/fun/inv_logit.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/log1p.hpp>
#include <stan/math/prim/fun/sum.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/select.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>

namespace stan {
namespace math {

template <typename T_y, typename T_loc, typename T_scale,
          require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_y, T_loc, T_scale>* = nullptr>
inline return_type_t<T_y, T_loc, T_scale> logistic_lccdf(const T_y& y,
                                                         const T_loc& mu,
                                                         const T_scale& sigma) {
  using T_partials_return = partials_return_t<T_y, T_loc, T_scale>;
  using T_y_ref = ref_type_if_not_constant_t<T_y>;
  using T_mu_ref = ref_type_if_not_constant_t<T_loc>;
  using T_sigma_ref = ref_type_if_not_constant_t<T_scale>;
  static constexpr const char* function = "logistic_lccdf";
  check_consistent_sizes(function, "Random variable", y, "Location parameter",
                         mu, "Scale parameter", sigma);
  T_y_ref y_ref = y;
  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  decltype(auto) y_val = to_ref(as_value_column_array_or_scalar(y_ref));
  decltype(auto) mu_val = to_ref(as_value_column_array_or_scalar(mu_ref));
  decltype(auto) sigma_val = to_ref(as_value_column_array_or_scalar(sigma_ref));
  check_not_nan(function, "Random variable", y_val);
  check_finite(function, "Location parameter", mu_val);
  check_positive_finite(function, "Scale parameter", sigma_val);

  if (size_zero(y, mu, sigma)) {
    return 0;
  }

  auto ops_partials = make_partials_propagator(y_ref, mu_ref, sigma_ref);

  // Explicit return for extreme values
  // The gradients are technically ill-defined, but treated as zero
  scalar_seq_view<decltype(y_val)> y_vec(y_val);
  for (size_t i = 0; i < stan::math::size(y_val); i++) {
    if (y_vec[i] == NEGATIVE_INFTY) {
      return ops_partials.build(0.0);
    }
  }
  for (size_t i = 0; i < stan::math::size(y_val); i++) {
    if (y_vec[i] == INFTY) {
      return ops_partials.build(negative_infinity());
    }
  }

  const auto& inv_sigma
      = to_ref_if<is_any_autodiff_v<T_y, T_loc, T_scale>>(inv(sigma_val));
  const auto& z = to_ref_if<is_any_autodiff_v<T_y, T_loc, T_scale>>(
      (y_val - mu_val) * inv_sigma);
  // log(inv_logit(t)) = min(t, 0) - log1p(exp(-|t|)), +0 at t = inf;
  // for t < 1, log(1 + exp(-|t|)) is within 3 ulp and cheaper
  const auto log_inv_logit_t = [](const auto& t) {
    if (t < 1) {
      return (t < 0 ? t : 0.0) - log(1 + exp(t < 0 ? t : -t));
    }
    return 0.0 - log1p(exp(-t));
  };
  T_partials_return P;
  if constexpr (is_eigen<std::decay_t<decltype(z)>>::value) {
    P = sum((-z).unaryExpr(log_inv_logit_t));
  } else {
    P = log_inv_logit_t(-z);
  }

  if constexpr (is_any_autodiff_v<T_y, T_loc, T_scale>) {
    // d/dz log(1 - inv_logit(z)) = -inv_logit(z)
    // evaluated: the expression holds a temporary holder by reference
    const auto& dz = to_ref(inv_logit(z) * inv_sigma);
    if constexpr (is_autodiff_v<T_y>) {
      partials<0>(ops_partials) = -dz;
    }
    if constexpr (is_autodiff_v<T_loc>) {
      partials<1>(ops_partials) = dz;
    }
    if constexpr (is_autodiff_v<T_scale>) {
      // z = +-inf contributes 0, not inf * 0
      partials<2>(ops_partials) = select(abs(z) == INFTY, 0.0, z * dz);
    }
  }
  return ops_partials.build(P);
}

}  // namespace math
}  // namespace stan
#endif

#ifndef STAN_MATH_PRIM_PROB_MULTINOMIAL_LOGIT_GLM_LPMF_HPP
#define STAN_MATH_PRIM_PROB_MULTINOMIAL_LOGIT_GLM_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/isfinite.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <cmath>
#include <vector>

namespace stan {
namespace math {

/** \ingroup multivar_dists
 * Returns the log PMF of the Generalized Linear Model (GLM)
 * with Multinomial distribution and softmax (logit) link function.
 * Efficiently computes
 *   sum_n multinomial_logit_lpmf(y[n] | eta_n),  eta_n = x_n * beta + alpha_n
 * with analytically derived gradients.
 *
 * The category probabilities for instance n are
 *   p_nk = softmax(eta_n)_k = exp(eta_nk) / sum_k' exp(eta_nk').
 * The log PMF (including the multinomial coefficient) is
 *   log PMF = sum_n [ lgamma(S_n+1) - sum_k lgamma(y_nk+1)
 *                   + sum_k y_nk * log p_nk ]
 * where S_n = sum_k y_nk is the total count for instance n.
 *
 * @tparam T_x type of the design matrix; either a matrix (N x M) or a row
 * vector (1 x M) broadcast over all instances
 * @tparam T_alpha type of the intercept; either a row vector (1 x K) broadcast
 * over all instances, or a matrix (N x K) with per-instance intercepts.
 * Analogous to a scalar vs vector alpha in univariate GLMs: the K dimension
 * corresponds to the class (output) space, not the instance dimension.
 * @tparam T_beta type of the weight matrix (M x K)
 * @param y outcome count vectors: `y[n]` is a length-K vector of non-negative
 * integer counts for instance n; the counts need not sum to a fixed total
 * @param x design matrix (N x M) or row vector (1 x M) broadcast over all
 * instances
 * @param alpha intercept; row vector (1 x K) broadcast across all instances,
 * or matrix (N x K) with one bias row per instance
 * @param beta weight matrix (M x K)
 * @return log sum of multinomial log PMFs over all N instances
 * @throw std::domain_error if any element of x, beta, or alpha is infinite, or
 * if any count in y is negative
 * @throw std::invalid_argument if container sizes mismatch
 */
template <bool propto, typename T_x, typename T_alpha, typename T_beta,
          require_matrix_t<T_x>* = nullptr,
          require_matrix_t<T_alpha>* = nullptr,
          require_matrix_t<T_beta>* = nullptr>
inline return_type_t<T_x, T_alpha, T_beta> multinomial_logit_glm_lpmf(
    const std::vector<std::vector<int>>& y, const T_x& x, const T_alpha& alpha,
    const T_beta& beta) {
  using T_partials_return = partials_return_t<T_x, T_alpha, T_beta>;
  using Eigen::Array;
  using Eigen::Dynamic;
  using std::exp;
  using std::isfinite;
  using std::log;
  using T_x_ref = ref_type_if_not_constant_t<T_x>;
  using T_alpha_ref = ref_type_if_not_constant_t<T_alpha>;
  using T_beta_ref = ref_type_if_not_constant_t<T_beta>;
  constexpr int T_x_rows = T_x::RowsAtCompileTime;
  constexpr int T_alpha_rows = T_alpha::RowsAtCompileTime;
  // η is the same K-vector for every instance iff both x and α are
  // broadcast (1 row each); otherwise each instance has a distinct η_n.
  constexpr int eta_rows = (T_x_rows == 1 && T_alpha_rows == 1) ? 1 : Dynamic;
  constexpr bool eta_is_broadcast = (eta_rows == 1);
  // When x is broadcast but α is N×K, the β/x gradients are
  // ∂ℓ/∂β = xᵀ * Σ_n(δ_n) and ∂ℓ/∂x = Σ_n(δ_n) * βᵀ,
  // i.e. δ must be summed over instances before the outer product.
  constexpr bool sum_delta_for_x = (T_x_rows == 1 && !eta_is_broadcast);
  constexpr bool gradients_calc = is_any_autodiff_v<T_x, T_alpha, T_beta>;

  // N_instances: derived from x when available, then α, then y
  const size_t N_instances = T_x_rows != 1   ? x.rows()
                             : T_alpha_rows != 1 ? alpha.rows()
                                                 : y.size();
  const size_t N_attributes = x.cols();
  const size_t N_classes = beta.cols();

  static constexpr const char* function = "multinomial_logit_glm_lpmf";
  check_size_match(function, "Rows of outcome vectors", y.size(),
                   "number of instances", N_instances);
  check_size_match(function, "Columns of intercept", alpha.cols(),
                   "number of classes", N_classes);
  if constexpr (T_x_rows != 1 && T_alpha_rows != 1) {
    check_size_match(function, "Rows of intercept", alpha.rows(),
                     "rows of design matrix", x.rows());
  }
  check_size_match(function, "Columns of design matrix", N_attributes,
                   "rows of weight matrix", beta.rows());

  if (size_zero(y)) {
    return 0;
  }
  for (size_t n = 0; n < N_instances; ++n) {
    check_size_match(function, "Size of outcome vector", y[n].size(),
                     "number of classes", N_classes);
    check_nonnegative(function, "outcome counts", y[n]);
  }

  if constexpr (!include_summand<propto, T_x, T_alpha, T_beta>::value) {
    return 0;
  }

  T_x_ref x_ref = x;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;

  const auto& x_val = to_ref_if<is_autodiff_v<T_beta>>(value_of(x_ref));
  const auto& alpha_val = value_of(alpha_ref);
  const auto& beta_val = to_ref_if<is_autodiff_v<T_x>>(value_of(beta_ref));

  // Base linear predictor x*β before adding the intercept α.
  const auto x_beta = (x_val * beta_val).eval();

  // η (eta_rows × K): η_nk = (x*β)_nk + α_nk.
  // Shape is 1×K when both x and α broadcast, else N×K.
  const Array<T_partials_return, eta_rows, Dynamic> eta = [&]() {
    if constexpr (T_alpha_rows == 1) {
      // Broadcast α: η = x*β + α (α added to every row)
      return (x_beta.rowwise() + alpha_val).array().eval();
    } else if constexpr (T_x_rows == 1) {
      // Broadcast x: tile x*β to N rows, then add per-instance α
      return (x_beta.replicate(N_instances, 1) + alpha_val).array().eval();
    } else {
      return (x_beta + alpha_val).array().eval();
    }
  }();

  // Row-wise maximum of η, used to center the log-sum-exp computation.
  const Array<T_partials_return, eta_rows, 1> eta_max = eta.rowwise().maxCoeff();
  // Centered linear predictor: η_nk - max_k(η_nk).
  // Subtracting the row maximum keeps all exponents ≤ 1 and prevents overflow
  // while leaving softmax values unchanged (max cancels in numerator/denominator).
  const Array<T_partials_return, eta_rows, Dynamic> shifted_eta
      = eta.colwise() - eta_max;
  // exp(η_nk - max_k η_nk). Materialised only when gradients are needed;
  // otherwise it is evaluated lazily inside sum_exp_η and then discarded.
  auto&& exp_eta = to_ref_if<gradients_calc>(exp(shifted_eta));
  // Partition function (per row): Z_n = Σ_k exp(η_nk - max_k η_nk).
  const Array<T_partials_return, eta_rows, 1> sum_exp_eta
      = exp_eta.rowwise().sum();
  // Log category probabilities: log p_nk = η_nk - log Z_n - max_k η_nk
  //                                      = shifted_η_nk - log(Z_n).
  const Array<T_partials_return, eta_rows, Dynamic> log_softmax_eta
      = shifted_eta.colwise() - log(sum_exp_eta);

  // Observed count matrix y_mat (N×K) and per-instance totals S_n = Σ_k y_nk.
  const Array<double, Dynamic, Dynamic> y_mat
      = Array<double, Dynamic, Dynamic>::NullaryExpr(
          N_instances, N_classes,
          [&y](Eigen::Index n, Eigen::Index k) -> double { return y[n][k]; });
  // S_n is the sample size of the n-th multinomial draw.
  const Array<double, Dynamic, 1> S = y_mat.rowwise().sum();

  // When η is broadcast (eta_is_broadcast), all instances share the same
  // log-probabilities, so the sufficient statistic for the log-likelihood is
  // y_totals_k = Σ_n y_nk (1×K).  Otherwise the full N×K matrix is needed.
  // The else branch returns a reference to y_mat — no copy.
  const auto& y_obs = [&]() -> decltype(auto) {
    if constexpr (eta_is_broadcast)
      return y_mat.colwise().sum().eval();
    else
      return (y_mat);
  }();

  // Log-likelihood: Σ_{n,k} y_nk * log p_nk = <y_obs, log_softmax_eta>_F
  T_partials_return logp = (y_obs * log_softmax_eta).sum();
  if constexpr (include_summand<propto>::value) {
    // Multinomial coefficient: Σ_n [ lgamma(S_n+1) - Σ_k lgamma(y_nk+1) ]
    logp += lgamma(S + 1.0).sum() - lgamma(y_mat + 1.0).sum();
  }

  if (!isfinite(logp)) {
    check_finite(function, "Weight matrix", beta_ref);
    check_finite(function, "Intercept", alpha_ref);
    check_finite(function, "Matrix of independent variables", x_ref);
  }

  auto ops_partials = make_partials_propagator(x_ref, alpha_ref, beta_ref);
  if constexpr (gradients_calc) {
    // Category probabilities: p_nk = softmax(η_n)_k = exp_η_nk / Z_n.
    const auto softmax_eta = (exp_eta.colwise() / sum_exp_eta).eval();
    // Gradient of log-likelihood w.r.t. η:
    //   ∂ℓ/∂η_nk = y_nk - S_n * p_nk  =: δ_nk.
    // When eta_is_broadcast (shared η): δ is 1×K with y_totals and total S.
    // Otherwise: δ is N×K with per-instance y_nk and S_n.
    auto delta = [&]() {
      if constexpr (eta_is_broadcast) {
        return (y_obs.template cast<T_partials_return>()
                - S.sum() * softmax_eta)
            .eval();
      } else {
        return (y_obs.template cast<T_partials_return>()
                - softmax_eta.colwise() * S.template cast<T_partials_return>())
            .eval();
      }
    }();

    // When x is broadcast, ∂ℓ/∂β = xᵀ * (Σ_n δ_n) and
    // ∂ℓ/∂x = (Σ_n δ_n) * βᵀ, so δ is collapsed to 1×K first.
    // Otherwise ∂ℓ/∂β = Xᵀ * δ (N×K) and ∂ℓ/∂x = δ * βᵀ.
    const auto delta_mat = [&]() {
      if constexpr (sum_delta_for_x)
        return delta.colwise().sum().matrix().eval();
      else
        return delta.matrix().eval();
    }();

    if constexpr (is_autodiff_v<T_alpha>) {
      // ∂ℓ/∂α_k = Σ_n δ_nk  (broadcast α, 1×K result)
      // ∂ℓ/∂α_nk = δ_nk     (per-instance α, N×K result)
      if constexpr (T_alpha_rows == 1)
        partials<1>(ops_partials) = delta.colwise().sum();
      else
        partials<1>(ops_partials) = delta;
    }
    if constexpr (is_autodiff_v<T_beta>) {
      // ∂ℓ/∂β = Xᵀ * δ_mat  (M×K)
      partials<2>(ops_partials)
          = x_val.transpose().template cast<T_partials_return>() * delta_mat;
    }
    if constexpr (is_autodiff_v<T_x>) {
      // ∂ℓ/∂x = δ_mat * βᵀ  (N×M, or 1×M when broadcast)
      edge<0>(ops_partials).partials_ = delta_mat * beta_val.transpose();
    }
  }
  return ops_partials.build(logp);
}

template <typename T_x, typename T_alpha, typename T_beta>
inline return_type_t<T_x, T_alpha, T_beta> multinomial_logit_glm_lpmf(
    const std::vector<std::vector<int>>& y, const T_x& x, const T_alpha& alpha,
    const T_beta& beta) {
  return multinomial_logit_glm_lpmf<false>(y, x, alpha, beta);
}

}  // namespace math
}  // namespace stan
#endif

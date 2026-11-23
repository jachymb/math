#ifndef STAN_MATH_OPENCL_REV_KRONECKER_HPP
#define STAN_MATH_OPENCL_REV_KRONECKER_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/prim/kronecker.hpp>
#include <stan/math/opencl/rev/arena_type.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/adjoint_of.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/value_of.hpp>

namespace stan {
namespace math {
namespace internal {

/** \ingroup opencl
 * Add to `adj`, read as a column-major vector of length `n`, the sums of
 * `terms(t, s)` over `s < k` for each `t < n`. Each sum gets one work item
 * if there are at least as many sums as terms per sum (`rowwise_sum`), and
 * work groups otherwise (`colwise_sum`), so that a small `adj` is not
 * reduced by a few serial work items.
 *
 * @tparam T_adj type of the adjoint matrix
 * @tparam F type of the functor building the terms
 * @param[in, out] adj adjoint to increment
 * @param n number of entries of `adj`
 * @param k number of terms per entry
 * @param terms functor that takes the kernel generator index expressions `t`
 * and `s` and returns the expression of the terms
 */
template <typename T_adj, typename F>
inline void kronecker_add_sums(T_adj& adj, int n, int k, const F& terms) {
  const int rows = adj.rows();
  if (n >= k) {
    indexing(adj, row_index(n, 1) % rows, elt_divide(row_index(n, 1), rows))
        += rowwise_sum(terms(row_index(n, k), col_index(n, k)));
  } else {
    matrix_cl<double> sums
        = colwise_sum(terms(col_index(k, n), row_index(k, n)));
    while (sums.rows() > 1) {
      sums = colwise_sum(sums).eval();
    }
    indexing(adj, col_index(1, n) % rows, elt_divide(col_index(1, n), rows))
        += sums;
  }
}

}  // namespace internal

/** \ingroup opencl
 * Return the Kronecker product of two matrices on the OpenCL device, at least
 * one of which holds `var`s.
 *
 * With \f$G\f$ the adjoint of the result, \f$t = i + m j\f$ and
 * \f$s = k + p l\f$, the adjoints are
 * \f$\bar{A}_t \mathrel{+}= \sum_s G_{p i + k, q j + l} B_{kl}\f$ and
 * \f$\bar{B}_s \mathrel{+}= \sum_t G_{p i + k, q j + l} A_{ij}\f$.
 *
 * @tparam T_a type of the first matrix
 * @tparam T_b type of the second matrix
 * @param A first matrix (m x n)
 * @param B second matrix (p x q)
 * @return Kronecker product of `A` and `B` (mp x nq)
 */
template <
    typename T_a, typename T_b, require_any_var_t<T_a, T_b>* = nullptr,
    require_all_nonscalar_prim_or_rev_kernel_expression_t<T_a, T_b>* = nullptr>
inline var_value<matrix_cl<double>> kronecker(T_a&& A, T_b&& B) {
  arena_t<T_a> A_arena = std::forward<T_a>(A);
  arena_t<T_b> B_arena = std::forward<T_b>(B);
  return make_callback_var(
      kronecker(value_of(A_arena), value_of(B_arena)),
      [A_arena, B_arena](const vari_value<matrix_cl<double>>& res) mutable {
        if (res.adj().size() == 0) {
          return;
        }
        const int m = A_arena.rows();
        const int mn = m * A_arena.cols();
        const int p = B_arena.rows();
        const int q = B_arena.cols();
        const int pq = p * q;
        auto res_adj_at = [&res, m, p, q](const auto& t, const auto& s) {
          return indexing(
              res.adj(), (t.deep_copy() % m) * p + s.deep_copy() % p,
              elt_divide(t.deep_copy(), m) * q + elt_divide(s.deep_copy(), p));
        };
        if constexpr (is_autodiff_v<T_a>) {
          internal::kronecker_add_sums(
              adjoint_of(A_arena), mn, pq, [&](const auto& t, const auto& s) {
                return elt_multiply(
                    res_adj_at(t, s),
                    indexing(value_of(B_arena), s.deep_copy() % p,
                             elt_divide(s.deep_copy(), p)));
              });
        }
        if constexpr (is_autodiff_v<T_b>) {
          internal::kronecker_add_sums(
              adjoint_of(B_arena), pq, mn, [&](const auto& s, const auto& t) {
                return elt_multiply(
                    res_adj_at(t, s),
                    indexing(value_of(A_arena), t.deep_copy() % m,
                             elt_divide(t.deep_copy(), m)));
              });
        }
      });
}

}  // namespace math
}  // namespace stan

#endif
#endif

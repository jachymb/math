#ifndef STAN_MATH_OPENCL_PRIM_KRONECKER_HPP
#define STAN_MATH_OPENCL_PRIM_KRONECKER_HPP
#ifdef STAN_OPENCL

#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/** \ingroup opencl
 * Return the Kronecker product of two matrices on the OpenCL device:
 * entry \f$(r, c)\f$ of the result is
 * \f$A_{\lfloor r / p \rfloor, \lfloor c / q \rfloor} B_{r \bmod p, c \bmod
 * q}\f$, where \f$B\f$ is \f$p \times q\f$.
 *
 * @tparam T_a type of the first matrix
 * @tparam T_b type of the second matrix
 * @param A first matrix (m x n)
 * @param B second matrix (p x q)
 * @return kernel generator expression for the Kronecker product (mp x nq)
 */
template <typename T_a, typename T_b,
          require_all_kernel_expressions_and_none_scalar_t<T_a, T_b>* = nullptr>
inline auto kronecker(T_a&& A, T_b&& B) {
  const int p = B.rows();
  const int q = B.cols();
  const int rows = A.rows() * p;
  const int cols = A.cols() * q;
  return elt_multiply(
      indexing(std::forward<T_a>(A), elt_divide(row_index(rows, cols), p),
               elt_divide(col_index(rows, cols), q)),
      indexing(std::forward<T_b>(B), row_index(rows, cols) % p,
               col_index(rows, cols) % q));
}

}  // namespace math
}  // namespace stan

#endif
#endif

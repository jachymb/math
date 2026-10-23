#ifndef STAN_MATH_PRIM_FUN_SYMMETRIZE_FROM_UPPER_TRI_HPP
#define STAN_MATH_PRIM_FUN_SYMMETRIZE_FROM_UPPER_TRI_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Return a symmetric (Hermitian for complex matrices) matrix using elements
 * from the upper triangular part of the input matrix. For a complex matrix
 * the lower triangle is the conjugate transpose of the upper one and the
 * diagonal is the real part of the diagonal of the input.
 *
 * @tparam T type of elements in the matrix
 * @param m Matrix.
 * @throw std:invalid_argument if the matrix is not square.
 */
template <typename T, require_eigen_t<T>* = nullptr>
inline Eigen::Matrix<value_type_t<T>, Eigen::Dynamic, Eigen::Dynamic>
symmetrize_from_upper_tri(const T& m) {
  check_square("symmetrize_from_upper_tri", "m", m);
  if constexpr (is_complex<value_type_t<T>>::value) {
    Eigen::Matrix<value_type_t<T>, Eigen::Dynamic, Eigen::Dynamic> res
        = m.template selfadjointView<Eigen::Upper>();
    for (Eigen::Index i = 0; i < res.rows(); ++i) {
      res(i, i) = value_type_t<T>(res(i, i).real());
    }
    return res;
  } else {
    return m.template selfadjointView<Eigen::Upper>();
  }
}

}  // namespace math
}  // namespace stan

#endif

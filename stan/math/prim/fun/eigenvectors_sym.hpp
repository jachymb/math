#ifndef STAN_MATH_PRIM_FUN_EIGENVECTORS_SYM_HPP
#define STAN_MATH_PRIM_FUN_EIGENVECTORS_SYM_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Return the eigenvectors of the specified symmetric (Hermitian for complex
 * matrices) matrix, in the order of the eigenvalues returned by
 * eigenvalues_sym.
 *
 * @tparam EigMat type of the matrix
 * @param m Specified matrix.
 * @return Matrix whose columns are the eigenvectors of m.
 * @throw std::invalid_argument if m is not square and has nonzero size
 * @throw std::domain_error if m is not symmetric (Hermitian for complex
 * matrices)
 */
template <typename EigMat, require_eigen_t<EigMat>* = nullptr,
          require_not_st_var<EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, Eigen::Dynamic, Eigen::Dynamic>
eigenvectors_sym(EigMat&& m) {
  if (unlikely(m.size() == 0)) {
    return Eigen::Matrix<value_type_t<EigMat>, -1, -1>(0, 0);
  }
  using PlainMat = plain_type_t<EigMat>;
  decltype(auto) m_ref = to_ref(std::forward<EigMat>(m));
  check_symmetric("eigenvectors_sym", "m", m_ref);

  Eigen::SelfAdjointEigenSolver<PlainMat> solver(
      std::forward<decltype(m_ref)>(m_ref));
  return solver.eigenvectors();
}

}  // namespace math
}  // namespace stan
#endif

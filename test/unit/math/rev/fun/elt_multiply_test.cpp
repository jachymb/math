#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

TEST_F(AgradRev, elt_multiply_rvalue_operands_moved_onto_arena) {
  using stan::math::elt_multiply;
  using stan::math::var;
  using vec_v = Eigen::Matrix<var, Eigen::Dynamic, 1>;
  auto grad_sum = [](const auto& res) {
    stan::math::set_zero_all_adjoints();
    stan::math::sum(res).grad();
  };
  // a moved operand is kept alive on the var_alloc_stack_, a copy is not
  const auto& moved = stan::math::ChainableStack::instance_->var_alloc_stack_;
  Eigen::VectorXd a_val(3);
  a_val << 1.5, -2, 3;
  Eigen::VectorXd b_val(3);
  b_val << 0.5, 4, -1;
  vec_v a = a_val;
  vec_v b = b_val;
  Eigen::VectorXd expected = a_val.cwiseProduct(b_val);

  std::size_t n_moved = moved.size();
  vec_v res = elt_multiply(vec_v(a), vec_v(b));
  EXPECT_EQ(n_moved + 2, moved.size());
  EXPECT_MATRIX_FLOAT_EQ(expected, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(b_val, a.adj());
  EXPECT_MATRIX_FLOAT_EQ(a_val, b.adj());

  // the moved double operand is read in the reverse pass
  n_moved = moved.size();
  res = elt_multiply(vec_v(a), Eigen::VectorXd(b_val));
  EXPECT_MATRIX_FLOAT_EQ(expected, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(b_val, a.adj());
  res = elt_multiply(Eigen::VectorXd(b_val), vec_v(a));
  EXPECT_MATRIX_FLOAT_EQ(expected, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(b_val, a.adj());
  EXPECT_EQ(n_moved + 4, moved.size());

  // lvalues are copied and stay usable
  n_moved = moved.size();
  res = elt_multiply(a, b_val);
  res = elt_multiply(a, b);
  EXPECT_EQ(n_moved, moved.size());
  EXPECT_EQ(3, a.size());
  EXPECT_EQ(3, b.size());
}

TEST_F(AgradRev, elt_multiply_rvalue_operands_size_mismatch) {
  using stan::math::elt_multiply;
  using vec_v = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
  vec_v a = Eigen::VectorXd::Ones(3);
  vec_v b = Eigen::VectorXd::Ones(4);
  EXPECT_THROW(elt_multiply(vec_v(a), vec_v(b)), std::invalid_argument);
  EXPECT_THROW(elt_multiply(vec_v(a), Eigen::VectorXd::Ones(4)),
               std::invalid_argument);
}

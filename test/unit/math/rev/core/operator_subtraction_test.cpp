#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdexcept>

TEST_F(AgradRev, subtract_rvalue_operands_moved_onto_arena) {
  using stan::math::subtract;
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
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(3);
  vec_v a = a_val;
  vec_v b = b_val;
  var c = 2.5;

  std::size_t n_moved = moved.size();
  vec_v res = subtract(vec_v(a), vec_v(b));
  EXPECT_EQ(n_moved + 2, moved.size());
  EXPECT_MATRIX_FLOAT_EQ(a_val - b_val, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(ones, a.adj());
  EXPECT_MATRIX_FLOAT_EQ(-ones, b.adj());

  n_moved = moved.size();
  res = subtract(vec_v(a), b_val);
  EXPECT_MATRIX_FLOAT_EQ(a_val - b_val, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(ones, a.adj());
  res = subtract(b_val, vec_v(a));
  EXPECT_MATRIX_FLOAT_EQ(b_val - a_val, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(-ones, a.adj());
  res = subtract(vec_v(a), c);
  EXPECT_MATRIX_FLOAT_EQ(a_val.array() - 2.5, res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(ones, a.adj());
  EXPECT_FLOAT_EQ(-3, c.adj());
  res = subtract(c, vec_v(a));
  EXPECT_MATRIX_FLOAT_EQ(2.5 - a_val.array(), res.val());
  grad_sum(res);
  EXPECT_MATRIX_FLOAT_EQ(-ones, a.adj());
  EXPECT_FLOAT_EQ(3, c.adj());
  EXPECT_EQ(n_moved + 4, moved.size());

  // operator- forwards to subtract
  stan::math::var_value<Eigen::VectorXd> s(a_val);
  n_moved = moved.size();
  stan::math::var_value<Eigen::VectorXd> d = s - vec_v(b);
  EXPECT_EQ(n_moved + 1, moved.size());
  EXPECT_MATRIX_FLOAT_EQ(a_val - b_val, d.val());
  grad_sum(d);
  EXPECT_MATRIX_FLOAT_EQ(ones, s.adj());
  EXPECT_MATRIX_FLOAT_EQ(-ones, b.adj());

  // empty rvalues
  EXPECT_EQ(0, subtract(vec_v(), vec_v()).size());

  // lvalues are copied and stay usable
  n_moved = moved.size();
  res = subtract(a, b);
  EXPECT_EQ(n_moved, moved.size());
  EXPECT_EQ(3, a.size());
  EXPECT_EQ(3, b.size());
}

TEST_F(AgradRev, subtract_rvalue_operands_size_mismatch) {
  using stan::math::subtract;
  using vec_v = Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1>;
  vec_v a = Eigen::VectorXd::Ones(3);
  vec_v b = Eigen::VectorXd::Ones(4);
  EXPECT_THROW(subtract(vec_v(a), vec_v(b)), std::invalid_argument);
  EXPECT_THROW(subtract(vec_v(a), Eigen::VectorXd::Ones(4)),
               std::invalid_argument);
  EXPECT_THROW(subtract(Eigen::VectorXd::Ones(4), vec_v(a)),
               std::invalid_argument);
}

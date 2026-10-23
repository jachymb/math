#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <limits>

namespace {
constexpr double inf = std::numeric_limits<double>::infinity();
}  // namespace

// lmultiply(0, b) is identically 0, so its partial with respect to b is 0
// at a = 0 instead of the 0/0 = NaN of a/b. The derivative of x * log(x)
// at x = 0 is then log(0) + 0 = -inf, its limit from above.
TEST_F(AgradRev, lmultiply_derivative_at_zero_scalar) {
  using stan::math::lmultiply;
  using stan::math::var;
  {
    var x = 0;
    lmultiply(x, x).grad();
    EXPECT_EQ(-inf, x.adj());
  }
  stan::math::recover_memory();
  {
    var a = 0;
    var b = 0;
    lmultiply(a, b).grad();
    EXPECT_EQ(-inf, a.adj());
    EXPECT_EQ(0, b.adj());
  }
  stan::math::recover_memory();
  {
    var b = 0;
    lmultiply(0.0, b).grad();
    EXPECT_EQ(0, b.adj());
  }
}

TEST_F(AgradRev, lmultiply_derivative_at_zero_var_matrix) {
  using stan::math::lmultiply;
  using stan::math::sum;
  using stan::math::var;
  using stan::math::var_value;
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd minf = Eigen::VectorXd::Constant(2, -inf);
  {
    var_value<Eigen::VectorXd> a(zero);
    var_value<Eigen::VectorXd> b(zero);
    sum(lmultiply(a, b)).grad();
    EXPECT_MATRIX_EQ(minf, a.adj());
    EXPECT_MATRIX_EQ(zero, b.adj());
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> b(zero);
    sum(lmultiply(zero, b)).grad();
    EXPECT_MATRIX_EQ(zero, b.adj());
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> a(zero);
    var b = 0;
    sum(lmultiply(a, b)).grad();
    EXPECT_MATRIX_EQ(minf, a.adj());
    EXPECT_EQ(0, b.adj());
  }
  stan::math::recover_memory();
  {
    var a = 0;
    var_value<Eigen::VectorXd> b(zero);
    sum(lmultiply(a, b)).grad();
    EXPECT_EQ(-inf, a.adj());
    EXPECT_MATRIX_EQ(zero, b.adj());
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> b(zero);
    sum(lmultiply(0.0, b)).grad();
    EXPECT_MATRIX_EQ(zero, b.adj());
  }
}

// A result element with zero adjoint contributes nothing to either argument,
// even where log(b) or a / b is infinite; without this every second
// derivative through b = 0 would be NaN (0 * log(0), 0 * a / 0).
TEST_F(AgradRev, lmultiply_zero_adjoint_at_zero_var_matrix) {
  using stan::math::elt_multiply;
  using stan::math::lmultiply;
  using stan::math::sum;
  using stan::math::var;
  using stan::math::var_value;
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd one = Eigen::VectorXd::Ones(2);
  Eigen::VectorXd w(2);
  w << 1, 0;
  {
    var_value<Eigen::VectorXd> a(one);
    var_value<Eigen::VectorXd> b(zero);
    sum(elt_multiply(w, lmultiply(a, b))).grad();
    EXPECT_EQ(-inf, a.adj()(0));
    EXPECT_EQ(0, a.adj()(1));
    EXPECT_EQ(inf, b.adj()(0));
    EXPECT_EQ(0, b.adj()(1));
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> b(zero);
    sum(elt_multiply(w, lmultiply(one, b))).grad();
    EXPECT_EQ(inf, b.adj()(0));
    EXPECT_EQ(0, b.adj()(1));
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> a(one);
    sum(elt_multiply(w, lmultiply(a, zero))).grad();
    EXPECT_EQ(-inf, a.adj()(0));
    EXPECT_EQ(0, a.adj()(1));
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> a(one);
    var b = 0;
    sum(elt_multiply(w, lmultiply(a, b))).grad();
    EXPECT_EQ(-inf, a.adj()(0));
    EXPECT_EQ(0, a.adj()(1));
    EXPECT_EQ(inf, b.adj());
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> a(one);
    var b = 0;
    sum(elt_multiply(zero, lmultiply(a, b))).grad();
    EXPECT_EQ(0, a.adj()(0));
    EXPECT_EQ(0, b.adj());
  }
  stan::math::recover_memory();
  {
    var_value<Eigen::VectorXd> a(one);
    sum(elt_multiply(w, lmultiply(a, 0.0))).grad();
    EXPECT_EQ(-inf, a.adj()(0));
    EXPECT_EQ(0, a.adj()(1));
  }
  stan::math::recover_memory();
  {
    Eigen::VectorXd b_val(2);
    b_val << 1, 0;
    var a = 1;
    var_value<Eigen::VectorXd> b(b_val);
    sum(elt_multiply(w, lmultiply(a, b))).grad();
    EXPECT_EQ(0, a.adj());
    EXPECT_EQ(1, b.adj()(0));
    EXPECT_EQ(0, b.adj()(1));
  }
  stan::math::recover_memory();
  {
    Eigen::VectorXd b_val(2);
    b_val << 1, 0;
    var a = 1;
    sum(elt_multiply(w, lmultiply(a, b_val))).grad();
    EXPECT_EQ(0, a.adj());
  }
  stan::math::recover_memory();
  {
    Eigen::VectorXd b_val(2);
    b_val << 1, 0;
    var_value<Eigen::VectorXd> b(b_val);
    sum(elt_multiply(w, lmultiply(2.0, b))).grad();
    EXPECT_EQ(2, b.adj()(0));
    EXPECT_EQ(0, b.adj()(1));
  }
}

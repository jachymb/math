#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST_F(AgradRev, Rev_to_arena_scalar_test) {
  int a = 2;
  auto b = stan::math::to_arena(a);
  EXPECT_EQ(b, a);
  EXPECT_TRUE((std::is_same<decltype(a), decltype(b)>::value));
  double a2 = 2;
  auto b2 = stan::math::to_arena(a2);
  EXPECT_EQ(b2, a2);
  EXPECT_TRUE((std::is_same<decltype(a2), decltype(b2)>::value));
}

TEST_F(AgradRev, Rev_to_arena_std_vector_test) {
  std::vector<int> a{1, 2};
  auto b = stan::math::to_arena(a);
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(a[i], b[i]);
  }
  EXPECT_FALSE((std::is_same<decltype(a), decltype(b)>::value));

  auto c = stan::math::to_arena(b);
  EXPECT_EQ(b.size(), c.size());
  EXPECT_EQ(b.data(), c.data());
}

TEST_F(AgradRev, Rev_to_arena_col_vector_test) {
  Eigen::VectorXd a(2);
  a << 1, 2;
  auto b = stan::math::to_arena(a);
  EXPECT_MATRIX_EQ(a, b);
  EXPECT_FALSE((std::is_same<decltype(a), decltype(b)>::value));
  auto c = stan::math::to_arena(b);
  EXPECT_EQ(b.size(), c.size());
  EXPECT_EQ(b.data(), c.data());
}

TEST_F(AgradRev, Rev_to_arena_row_vector_test) {
  Eigen::RowVectorXd a(2);
  a << 1, 2;
  auto b = stan::math::to_arena(a);
  EXPECT_MATRIX_EQ(a, b);
  EXPECT_FALSE((std::is_same<decltype(a), decltype(b)>::value));
  auto c = stan::math::to_arena(b);
  EXPECT_EQ(b.size(), c.size());
  EXPECT_EQ(b.data(), c.data());
}

TEST_F(AgradRev, Rev_to_arena_matrix_test) {
  Eigen::MatrixXd a(2, 2);
  a << 1, 2, 3, 4;
  auto b = stan::math::to_arena(a);
  EXPECT_MATRIX_EQ(a, b);
  EXPECT_FALSE((std::is_same<decltype(a), decltype(b)>::value));
  auto c = stan::math::to_arena(b);
  EXPECT_EQ(b.size(), c.size());
  EXPECT_EQ(b.data(), c.data());
}

TEST_F(AgradRev, Rev_to_arena_rvalue_matrix_test) {
  const auto& moved = stan::math::ChainableStack::instance_->var_alloc_stack_;
  Eigen::VectorXd a(3);
  a << 1, 2, 3;
  Eigen::VectorXd expected = a;
  Eigen::VectorXd b = a;

  // an rvalue keeps its memory, which is kept alive until recover_memory()
  const double* a_data = a.data();
  std::size_t n_moved = moved.size();
  auto a_arena = stan::math::to_arena(std::move(a));
  EXPECT_EQ(a_data, a_arena.data());
  EXPECT_EQ(n_moved + 1, moved.size());
  EXPECT_MATRIX_EQ(expected, a_arena);

  const double* b_data = b.data();
  auto b_arena = stan::math::to_arena_if<true>(std::move(b));
  EXPECT_EQ(b_data, b_arena.data());
  EXPECT_EQ(n_moved + 2, moved.size());
  EXPECT_MATRIX_EQ(expected, b_arena);

  // an lvalue is copied and left unchanged
  Eigen::VectorXd c = expected;
  auto c_arena = stan::math::to_arena(c);
  EXPECT_NE(c.data(), c_arena.data());
  EXPECT_EQ(n_moved + 2, moved.size());
  EXPECT_MATRIX_EQ(expected, c);
  EXPECT_MATRIX_EQ(expected, c_arena);

  // an empty rvalue is moved as well
  auto e_arena = stan::math::to_arena(Eigen::VectorXd());
  EXPECT_EQ(0, e_arena.size());
  EXPECT_EQ(n_moved + 3, moved.size());
}

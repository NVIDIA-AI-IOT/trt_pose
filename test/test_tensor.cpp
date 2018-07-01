#include "gtest/gtest.h"

#include "../src/tensor.h"

TEST(tensor2_index, Valid) {
  tensor2_t t = { {3, 3}, {3, 1} };

  ASSERT_EQ(0, tensor2_index(&t, 0, 0));
  ASSERT_EQ(1, tensor2_index(&t, 0, 1));
  ASSERT_EQ(3, tensor2_index(&t, 1, 0));
  ASSERT_EQ(4, tensor2_index(&t, 1, 1));
  ASSERT_EQ(6, tensor2_index(&t, 2, 0));
  ASSERT_EQ(7, tensor2_index(&t, 2, 1));
}

TEST(tensor3_index, Valid) {
}

TEST(tensor4_index, Valid) {
}

TEST(tensor2_set_sizes, Valid) {
  tensor2_t t;
  tensor2_set_sizes(&t, 3, 2);
  ASSERT_EQ(t.sizes[0], 3);
  ASSERT_EQ(t.sizes[1], 2);
}

TEST(tensor3_set_sizes, Valid) {
  tensor3_t t;
  tensor3_set_sizes(&t, 3, 2, 1);
  ASSERT_EQ(t.sizes[0], 3);
  ASSERT_EQ(t.sizes[1], 2);
  ASSERT_EQ(t.sizes[2], 1);
}

TEST(tensor4_set_sizes, Valid) {
  tensor4_t t;
  tensor4_set_sizes(&t, 4, 3, 2, 1);
  ASSERT_EQ(t.sizes[0], 4);
  ASSERT_EQ(t.sizes[1], 3);
  ASSERT_EQ(t.sizes[2], 2);
  ASSERT_EQ(t.sizes[3], 1);
}

TEST(tensor2_set_linear_strides, Valid) {
  tensor2_t t;
  tensor2_set_sizes(&t, 3, 2);
  tensor2_set_linear_strides(&t);
  ASSERT_EQ(t.strides[0], 2);
  ASSERT_EQ(t.strides[1], 1);
}

TEST(tensor3_set_linear_strides, Valid) {
  tensor3_t t;
  tensor3_set_sizes(&t, 4, 3, 2);
  tensor3_set_linear_strides(&t);
  ASSERT_EQ(t.strides[0], 6);
  ASSERT_EQ(t.strides[1], 2);
  ASSERT_EQ(t.strides[2], 1);
}

TEST(tensor4_set_linear_strides, Valid) {
  tensor4_t t;
  tensor4_set_sizes(&t, 5, 4, 3, 2);
  tensor4_set_linear_strides(&t);
  ASSERT_EQ(t.strides[0], 24);
  ASSERT_EQ(t.strides[1], 6);
  ASSERT_EQ(t.strides[2], 2);
  ASSERT_EQ(t.strides[3], 1);
}

TEST(tensor2_get_size, Valid) {
  tensor2_t t;
  tensor2_set_sizes(&t, 3, 2);
  tensor2_set_linear_strides(&t);

  ASSERT_EQ(6, tensor2_get_size(&t));
}

TEST(tensor3_get_size, Valid) {
  tensor3_t t;
  tensor3_set_sizes(&t, 4, 3, 2);
  tensor3_set_linear_strides(&t);
  ASSERT_EQ(24, tensor3_get_size(&t));
}

TEST(tensor4_get_size, Valid) {
  tensor4_t t;
  tensor4_set_sizes(&t, 5, 4, 3, 2);
  tensor4_set_linear_strides(&t);
  ASSERT_EQ(120, tensor4_get_size(&t));
}

TEST(tensor2_transpose, ValidSize) {
  tensor2_t a;
  tensor2_set_sizes(&a, 3, 2);
  tensor2_t b = tensor2_transpose(&a);
  ASSERT_EQ(6, tensor2_get_size(&b)); 
};

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

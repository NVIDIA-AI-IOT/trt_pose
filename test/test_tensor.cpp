#include "gtest/gtest.h"

#include "../src/tensor.h"

TEST(tensor2, Valid) {
  tensor2 t = { {3, 3}, {3, 1} };

  ASSERT_EQ(0, tensor2_index(&t, 0, 0));
  ASSERT_EQ(1, tensor2_index(&t, 0, 1));
  ASSERT_EQ(3, tensor2_index(&t, 1, 0));
  ASSERT_EQ(4, tensor2_index(&t, 1, 1));
  ASSERT_EQ(6, tensor2_index(&t, 2, 0));
  ASSERT_EQ(7, tensor2_index(&t, 2, 1));
}

TEST(tensor3, Valid) {
}

TEST(tensor4, Valid) {
}



int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

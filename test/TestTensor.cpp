#include "gtest/gtest.h"

#include <memory>

#include "../src/Tensor.h"
#include "../src/TensorLayout.h"

template<typename T>
void AllEqual(const std::vector<T> &a, const std::vector<T> &b)
{
  ASSERT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    ASSERT_EQ(a[i], b[i]);
  }
}

TEST(TestGTest, Hello) {
  ASSERT_STREQ("Hello", "Hello");
}

TEST(DefaultStrides, OneDimension) {
  AllEqual<int64_t>({ 1 }, DefaultStrides({ 1 }));
  AllEqual<int64_t>({ 1 }, DefaultStrides({ 2 }));
  AllEqual<int64_t>({ 1 }, DefaultStrides({ 4 }));
  AllEqual<int64_t>({ 1 }, DefaultStrides({ 16 }));
}

TEST(DefaultStrides, TwoDimension) {
  AllEqual<int64_t>({ 1, 1 }, DefaultStrides({ 1, 1 }));
  AllEqual<int64_t>({ 2, 1 }, DefaultStrides({ 1, 2 }));
  AllEqual<int64_t>({ 1, 1 }, DefaultStrides({ 2, 1 }));
  AllEqual<int64_t>({ 4, 1 }, DefaultStrides({ 8, 4 }));
}

TEST(DefaultStrides, ThreeDimension) {
  AllEqual<int64_t>({ 1, 1, 1 }, DefaultStrides({ 1, 1, 1 }));
  AllEqual<int64_t>({ 2, 1, 1 }, DefaultStrides({ 3, 2, 1 }));
  AllEqual<int64_t>({ 6, 2, 1 }, DefaultStrides({ 4, 3, 2 }));
}


int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

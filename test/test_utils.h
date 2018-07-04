#include "gtest/gtest.h"

template<typename T>
void assert_all_equal(T *a, T *b, size_t size) {
  for (size_t i = 0; i < size; i++) {
    ASSERT_EQ(a[i], b[i]);
  }
}

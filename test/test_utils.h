#pragma once

#include "gtest/gtest.h"

void AllFloatEqual(float *a, float *b, size_t N)
{
  for (size_t i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(a[i], b[i]);
  }
}

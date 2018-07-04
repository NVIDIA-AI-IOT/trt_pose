#define EXCLUDE_MAIN

#include "gtest/gtest.h"
#include "test_peak_local_max.cu"

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

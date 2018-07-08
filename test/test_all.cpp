#define EXCLUDE_MAIN

#include "gtest/gtest.h"
#include "test_peak_local_max.cpp"
#include "test_munkres.cpp"
#include "test_connected_components.cpp"

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

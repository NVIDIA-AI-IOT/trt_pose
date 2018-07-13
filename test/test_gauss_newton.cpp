#include "gtest/gtest.h"

#include "../src/Matrix.hpp"
#include "../src/GaussNewton.hpp"

TEST(gauss_newton_update, valid)
{
  const int n = 4;
  const int m = 3;

  float param_data[m] = { 1.0f, 2.0f, 3.0f };
  float residual_data[n] = { 1.0f, 2.0f, 3.0f, 4.0f };
  float jacobian_data[n * m] = {
    1, 0, 0,
    0.5, 0, 0.5,
    1, 0, 0,
    0.5, 0.5, 0.5
  }; 

  Matrix<float> param(param_data, m, 1);
  Matrix<float> residual(residual_data, n, 1);
  Matrix<float> jacobian(jacobian_data, n, m);

  gaussNewtonUpdate(param, residual, jacobian);
  ASSERT_NEAR(-1, param.at(0, 0), 0.001);
  ASSERT_NEAR(-2, param.at(1, 0), 0.001);
  ASSERT_NEAR(1, param.at(2, 0), 0.001);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

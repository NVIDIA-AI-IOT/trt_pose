#include "GaussianFit.hpp"
#include "Gaussian.hpp"

#include <cmath>

void computeResidualJacobian(
    const std::pair<int, int> peak, 
    const Matrix<float> cmap, 
    const Matrix<float> param,
    Matrix<float> residual, 
    Matrix<float> jacobian,
    int wsize)
{
  int wsize_half = wsize / 2;

  // compute bounds
  int i_min = (peak.first - wsize_half) >= 0 ? (peak.first - 1) : 0;
  int j_min = (peak.second - wsize_half) >= 0 ? (peak.second - 1) : 0;
  int i_max = (peak.first + wsize_half) < cmap.nrows ? (peak.first + wsize_half) : cmap.nrows;
  int j_max = (peak.second + wsize_half) < cmap.ncols ? (peak.second + wsize_half) : cmap.ncols;

  residual.fill_zero();
  jacobian.fill_zero();

  int sample = 0;

  for (int i = i_min; i <= i_max; i++)
  {
    float i_diff = i - param.at(0, 0);
    float i_diff_2 = i_diff * i_diff;
    for (int j = j_min; j <= j_max; j++)
    {
      float j_diff = j - param.at(1, 0);
      float j_diff_2 = j_diff * j_diff;
      float exp_val = expf(-(i_diff_2 + j_diff_2) / (2.0f * param.at(3, 0)));
      float ij_coef = -param.at(2, 0) * exp_val / param.at(3, 0);

      *residual.at_(sample, 0) = cmap.at(i, j) - param.at(2, 0) * exp_val;

      *jacobian.at_(sample, 0) = ij_coef * i_diff;
      *jacobian.at_(sample, 1) = ij_coef * j_diff;
      *jacobian.at_(sample, 2) = - exp_val;
      *jacobian.at_(sample, 3) = ij_coef * (i_diff_2 + j_diff_2) / (2.0f * param.at(3, 0));
      sample++;
    }
  }
}


Gaussian gaussianFit(const std::pair<int, int> peak, const Matrix<float> cmap, int wsize, int niter)
{
  Matrix<float> param(NPARAM, 1);
  Matrix<float> residual(wsize * wsize, 1);
  Matrix<float> jacobian(wsize * wsize, NPARAM);

  *param.at_(0, 0) = peak.first;
  *param.at_(1, 0) = peak.second;
  *param.at_(2, 0) = DEFAULT_ALPHA;
  *param.at_(3, 0) = DEFAULT_SIGMA2;

  for (int i = 0; i < niter; i++)
  {
    computeResidualJacobian(peak, cmap, param, residual, jacobian, wsize);
    gaussNewtonUpdate(param, residual, jacobian);
  }

  Gaussian result;
  result.mean_i = param.at(0, 0);
  result.mean_j = param.at(1, 0);
  result.alpha = param.at(2, 0);
  result.sigma2 = param.at(3, 0);

  return result;
}

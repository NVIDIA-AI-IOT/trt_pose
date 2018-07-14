#include "GaussNewton.hpp"

#include "lapacke.h"
#include "cblas.h"

void gaussNewtonUpdate(Matrix<float> param, Matrix<float> &residual, Matrix<float> &jacobian)
{
  float alpha = 1.0f;
  float beta = 0.0f;
  Matrix<float> tmp_mxm_0(jacobian.ncols, jacobian.ncols);
  Matrix<float> tmp_mxm_1(jacobian.ncols, jacobian.ncols);
  Matrix<float> tmp_mxn(jacobian.ncols, jacobian.nrows);

  // J^T * J
  cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, 
      CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans,
      jacobian.ncols, jacobian.ncols, jacobian.nrows,
      alpha,
      jacobian.dataMutable(),
      jacobian.ncols,
      jacobian.dataMutable(),
      jacobian.ncols,
      beta,
      tmp_mxm_0.dataMutable(),
      tmp_mxm_0.ncols
  );

  // inv( " )
  tmp_mxm_1.fill_identity();
  LAPACKE_sposv(LAPACK_ROW_MAJOR, 'U', jacobian.ncols, jacobian.ncols, tmp_mxm_0.dataMutable(),
   tmp_mxm_0.ncols, tmp_mxm_1.dataMutable(), tmp_mxm_1.ncols);   

  // " * J^T
  cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor,
      CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
      jacobian.ncols, jacobian.nrows, jacobian.ncols,
      alpha,
      tmp_mxm_1.dataMutable(),
      tmp_mxm_1.ncols,
      jacobian.dataMutable(),
      jacobian.ncols,
      beta,
      tmp_mxn.dataMutable(),
      tmp_mxn.ncols
  );

  // param -= " * R
  beta = 1.0f;
  alpha = -1.0f;
  cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor,
      CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
      param.nrows, param.ncols, tmp_mxn.ncols,
      alpha,
      tmp_mxn.dataMutable(),
      tmp_mxn.ncols,
      residual.dataMutable(),
      residual.ncols,
      beta,
      param.dataMutable(),
      param.ncols
  );
}

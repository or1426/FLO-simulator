#include "wrappers.h"

int matmul_square_complex(std::vector<std::complex<double> > const& A, std::vector<std::complex<double> > const& B, std::vector<std::complex<double> > &C, int n){
  std::complex<double> ONE = 1;
  std::complex<double> ZERO = 0;
  cblas_zgemm(CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              &ONE,            // value to scale the values with
              &A[0],
              n,
              &B[0],
              n,
              &ZERO,            // value use to scale output
              &C[0],
              n);           //   for row major
  return 0;
}

std::vector<std::complex<double> > matmul_square_complex(std::vector<std::complex<double> > const& A, std::vector<std::complex<double> > const& B,  int n){
  std::vector<std::complex<double> > C(A.size());
  std::complex<double> ONE = 1;
  std::complex<double> ZERO = 0;
  cblas_zgemm(CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              &ONE,
              &A[0],
              n,
              &B[0],
              n,
              &ZERO,
              &C[0],
              n);
  return C;

}

std::vector<std::complex<double> > matmul_square_complex(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, std::vector<std::complex<double> > const& A, std::vector<std::complex<double> > const& B,  int n){
  std::vector<std::complex<double> > C(A.size());
  std::complex<double> ONE = 1;
  std::complex<double> ZERO = 0;
  cblas_zgemm(CblasColMajor,
              transA,
              transB,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              &ONE,
              &A[0],
              n,
              &B[0],
              n,
              &ZERO,
              &C[0],
              n);
  return C;
}

std::vector<double> matmul_square_double(const std::vector<double> &A, const std::vector<double>  &B,  int n){
  std::vector<double> C(A.size());
  double ONE = 1;
  double ZERO = 0;
  cblas_dgemm(CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              ONE,
              &A[0],
              n,
              &B[0],
              n,
              ZERO,
              &C[0],
              n);
  return C;
}

int matmul_square_double(std::vector<double> const& A, std::vector<double> const& B,  std::vector<double>  &C, int n){
  double ONE = 1;
  double ZERO = 0;
  cblas_dgemm(CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              ONE,
              &A[0],
              n,
              &B[0],
              n,
              ZERO,
              &C[0],
              n);
  return 0;
}

int matmul_square_double(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, std::vector<double> const& A, std::vector<double>  const& B,  std::vector<double>  &C, int n){
  double ONE = 1;
  double ZERO = 0;
  cblas_dgemm(CblasColMajor,
              transA,
              transB,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              ONE,
              &A[0],
              n,
              &B[0],
              n,
              ZERO,
              &C[0],
              n);
  return 0;
}

std::vector<double> matmul_square_double(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, std::vector<double> const& A, std::vector<double> const& B,   int n){
  std::vector<double>  C(n*n);
  double ONE = 1;
  double ZERO = 0;
  cblas_dgemm(CblasColMajor,
              transA,
              transB,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              ONE,
              &A[0],
              n,
              &B[0],
              n,
              ZERO,
              &C[0],
              n);
  return C;
}


int matrix_conjugate_inplace_complex(std::vector<std::complex<double> > &A,std::vector<std::complex<double> > const& B,int n){
  //A <- B A B^T
  std::vector<std::complex<double> > scratch(A.size());
  std::complex<double> ONE = 1;
  std::complex<double> ZERO = 0;
  cblas_zgemm(CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              &ONE,
              &B[0],
              n,
              &A[0],
              n,
              &ZERO,
              &scratch[0],
              n);
  cblas_zgemm(CblasColMajor,
              CblasNoTrans,
              CblasTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of second input matrix
              &ONE,
              &scratch[0],
              n,
              &B[0],
              n,
              &ZERO,
              &A[0],
              n);

  return 0;
}

int matrix_conjugate_inplace_complex(std::vector<std::complex<double> > &A,std::vector<std::complex<double> > const& B,int n, CBLAS_TRANSPOSE trans){
  //A <- B A B^T
  std::vector<std::complex<double> > scratch(A.size());
  std::complex<double> ONE = 1;
  std::complex<double> ZERO = 0;
  cblas_zgemm(CblasColMajor,
              trans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              &ONE,
              &B[0],
              n,
              &A[0],
              n,
              &ZERO,
              &scratch[0],
              n);
  if(trans == CblasNoTrans){
    trans = CblasTrans;
  }else{
    trans = CblasNoTrans;
  }
  cblas_zgemm(CblasColMajor,
              CblasNoTrans,
              trans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              &ONE,
              &scratch[0],
              n,
              &B[0],
              n,
              &ZERO,
              &A[0],
              n);

  return 0;
}


int matrix_conjugate_inplace_double(std::vector<double> &A,std::vector<double> const& B,int n){
  //A <- B A B^T
  double ONE = 1;
  double ZERO = 0;
  std::vector<double> scratch(A.size());
  cblas_dgemm(CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              ONE,
              &B[0],
              n,
              &A[0],
              n,
              ZERO,
              &scratch[0],
              n);
  cblas_dgemm(CblasColMajor,
              CblasNoTrans,
              CblasTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              ONE,
              &scratch[0],
              n,
              &B[0],
              n,
              ZERO,
              &A[0],
              n);

  return 0;
}


int matrix_conjugate_inplace_double(std::vector<double> &A,std::vector<double> const& B,int n, CBLAS_TRANSPOSE trans){
  //A <- B A B^T
  double ONE = 1;
  double ZERO = 0;
  std::vector<double> scratch(A.size());
  cblas_dgemm(CblasColMajor,
              trans,
              CblasNoTrans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              ONE,
              &B[0],
              n,
              &A[0],
              n,
              ZERO,
              &scratch[0],
              n);
  if(trans == CblasNoTrans){
    trans = CblasTrans;
  }else{
    trans = CblasNoTrans;
  }

  cblas_dgemm(CblasColMajor,
              CblasNoTrans,
              trans,
              n,            // rows in the first input matrix
              n,            // columns in the second input matrix
              n,            // columns of first input matrix, rows of   second input matrix
              ONE,
              &scratch[0],
              n,
              &B[0],
              n,
              ZERO,
              &A[0],
              n);

  return 0;
}


void matrix_add_block(std::vector<std::complex<double> > &A, std::vector<double> B, std::complex<double> scaling,
                      int Adim, int Bdim,
                      int AstartX, int AstartY, int BstartX, int BstartY, int block_width, int block_height){
  for(int i = 0; i < block_height; i++)
  {
    for(int j = 0; j < block_width; j++){
      A[dense_fortran(i+1+AstartX, j+1+AstartY, Adim)] += B[dense_fortran(i+1+BstartX, j+1+BstartY, Bdim)]*scaling;
    }
  }
}

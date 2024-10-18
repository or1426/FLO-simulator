#ifndef FLO_FID_WRAPPERS_H
#define FLO_FID_WRAPPERS_H
#include <iostream>
#include <vector>
#include <cblas.h>
#include <complex.h>
#include <cmath>
#include "pfapack.h" //we only need this for dense_fortran I think

template<typename T>
void print_fortran(std::vector<T> A, int n){
  for(int i = 1; i <= n; i++){
    for(int j = 1; j <= n; j++){
      std::cout << A[dense_fortran(i,j,n)] << " ";
    }
    std::cout << std::endl;
  }
}

int matmul_square_complex(std::vector<std::complex<double> > const& A, std::vector<std::complex<double> > const& B, std::vector<std::complex<double> > &C, int n);
std::vector<std::complex<double> > matmul_square_complex(std::vector<std::complex<double> > const& A, std::vector<std::complex<double> > const& B,  int n);
std::vector<std::complex<double> > matmul_square_complex(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, std::vector<std::complex<double> > const& A, std::vector<std::complex<double> > const& B,  int n);
std::vector<double> matmul_square_double(std::vector<double> const& A, std::vector<double>  const& B,  int n);
int matmul_square_double(std::vector<double> const& A, std::vector<double> const& B,  std::vector<double>  &C, int n);
int matmul_square_double(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, std::vector<double> const& A, std::vector<double> const& B,  std::vector<double>  &C, int n);
std::vector<double> matmul_square_double(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, std::vector<double> const& A, std::vector<double> const& B,   int n);
int matrix_conjugate_inplace_complex(std::vector<std::complex<double> > &A,std::vector<std::complex<double> > const& B,int n);
int matrix_conjugate_inplace_complex(std::vector<std::complex<double> > &A,std::vector<std::complex<double> > const& B,int n, CBLAS_TRANSPOSE trans);
int matrix_conjugate_inplace_double(std::vector<double> &A,std::vector<double> const& B,int n);
int matrix_conjugate_inplace_double(std::vector<double> &A,std::vector<double> const& B,int n, CBLAS_TRANSPOSE trans);
void matrix_add_block(std::vector<std::complex<double> > &A, std::vector<double> B, std::complex<double> scaling, int Adim, int Bdim, int AstartX, int AstartY, int BstartX, int BstartY, int block_width, int block_height);
#endif

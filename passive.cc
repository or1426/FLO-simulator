#include "passive.h"

using namespace std::complex_literals;


DecomposedPassive PassiveFLO::decompose(){
    //we use lapack so we need some different types
  using cdouble = std::complex<double>; 
  const int32_t n = this->qubits;
  std::vector<cdouble> U(n*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      U[dense_fortran(i+1,j+1,n)] = this->R[dense_fortran(2*i+1,2*j+1, 2*n)] + (this->R[dense_fortran(2*i+1,2*j+2, 2*n)]*(1.i));
    }
  }
  std::vector<cdouble> eigenvectors(n*n);
  std::vector<cdouble>  eigenvalues(n);

  int32_t m;
  int32_t isuppz;

  /*
    ZGEES( JOBVS, SORT, SELECT, N, A, LDA, SDIM, W, VS, LDVS, WORK,
    LWORK, RWORK, BWORK, INFO )

    CHARACTER       JOBVS, SORT

    INTEGER         INFO, LDA, LDVS, LWORK, N, SDIM

    LOGICAL         BWORK( * )

    DOUBLE          PRECISION RWORK( * )

    COMPLEX*16    A( LDA, * ), VS( LDVS, * ),   W( * ), WORK( * )

    LOGICAL         SELECT

    EXTERNAL        SELECT
  */
  std::vector<double> rwork(n);
  int32_t info;
  int32_t MINUS_1 = -1;
  int sdim = 0;

  std::vector<cdouble> workopt(1);
  LAPACK_zgees("V", "N", NULL, &n, &U[0], &n, &sdim,
               &eigenvalues[0], &eigenvectors[0],
               &n, &workopt[0], &MINUS_1, &rwork[0], NULL, &info);


  int lwork = (int)workopt[0].real();
  std::vector<cdouble> work(lwork);
  LAPACK_zgees("V", "N", NULL, &n, &U[0], &n, &sdim,
               &eigenvalues[0], &eigenvectors[0],
               &n, &work[0], &lwork, &rwork[0], NULL, &info);


  std::vector<double> V(2*n*2*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      V[dense_fortran(2*j+1, 2*i+1, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].real();
      V[dense_fortran(2*j+2, 2*i+2, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].real();
      V[dense_fortran(2*j+1, 2*i+2, 2*n)] =-eigenvectors[dense_fortran(i+1, j+1, n)].imag();
      V[dense_fortran(2*j+2, 2*i+1, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].imag();
    }
  }

  //print_fortran(newR, 2*n);
  //we now compute the log of newR
  //which is block diagonal with 2x2 orthogonal blocks
  //so this is pretty easy
  std::vector<double> lambdas(n);
  double sum = 0;
  for(int i = 0; i < n; i++){
    lambdas[i] = -std::atan2((eigenvalues[i].imag()), (eigenvalues[i].real())); //TODO CHECK THIS!!!!!
    sum += lambdas[i]/2.;
  }
  std::complex<double> new_phase = std::exp(sum*1.i);

  //we might be wrong by a factor of minus 1 in new phase
  //if we are we have to add 2pi to lambda_0
  if(this->phase){
    if(abs(*(this->phase) - new_phase) > abs(*(this->phase) + new_phase)){
      lambdas[0] += 2*M_PI;
      new_phase = -new_phase;
    }
  }
  DecomposedPassive p;
  p.phase = new_phase;
  p.l = lambdas;
  p.R = V;


  return p;

}

PassiveFLO PassiveFLO::multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, PassiveFLO A, PassiveFLO B){
  //std::optional<std::complex<double> > phase;
  if(A.phase && B.phase){
    std::complex<double> phase = ((transA == CblasTrans) ? std::conj(*A.phase) : *A.phase) *
      ((transB == CblasTrans) ? std::conj(*B.phase) : *B.phase);
    return PassiveFLO(A.qubits, //we ignore the possibility they have different numbers of qubits
		      matmul_square_double(transB, transA, B.R, A.R, 2*A.qubits),
		      phase);

  }else{
    return PassiveFLO(A.qubits, //we ignore the possibility they have different numbers of qubits
		      matmul_square_double(transB, transA, B.R, A.R, 2*A.qubits));
  }
  

}

//DecomposedPassive decompose_passive_flo_unitary(std::vector<double> R,int qubits, std::complex<double> phase){
//}


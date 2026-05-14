#ifndef PASSIVE_FERMIONIC_LINEAR_OPTICS_H
#define PASSIVE_FERMIONIC_LINEAR_OPTICS_H
#include <vector>
#include <optional>
#include <complex.h>
#include <cmath>

#ifndef lapack_complex_double
#define lapack_complex_double std::complex<double>
#endif 
#include <lapack.h>

#include "wrappers.h"
#include <iomanip>
typedef struct DecomposedPassive{
  //R is the orthogonal matrix implementing a passive FLO unitary K such that
  //K exp(sum_j c_{2j}c_{2j+1} lambda_j/2) K^\dagger = U
  //while <0|U|0> = phase
  std::complex<double> phase;
  std::vector<double> l;
  std::vector<double> R;
} DecomposedPassive;


class PassiveFLO{
 public:
  int qubits;
  std::vector<double> R;
  std::optional<std::complex<double> > phase;//a passive flo may or may not know its phase
  //construct an nxn identity 
  PassiveFLO(int qubits){
    this->qubits = qubits;
    this->R = std::vector<double>(2*qubits*2*qubits,0.);
    this->phase = 1.;

    for(int i = 0; i < 2*qubits; i++){
      this->R[dense_fortran(i+1, i+1, 2*qubits)] = 1.;
    }
  }
  PassiveFLO(int qubits, std::vector<double> R){
    this->qubits = qubits;
    this->R = R;
    this->phase = std::nullopt;
  }
  PassiveFLO(int qubits, std::vector<double> R, std::complex<double> phase){
    this->qubits = qubits;
    this->R = R;
    this->phase = phase;
  }

  //note that the map from FLO unitaries to orthogonal matrices is an anti-homomorphism
  //this means that when you do K1 * K2, the orthogonal matrices multiply like R2 * R1
  static PassiveFLO multiply(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, PassiveFLO A, PassiveFLO B);
  static PassiveFLO multiply(PassiveFLO A, PassiveFLO B);
  DecomposedPassive decompose();
  
  //we generate a passive flo operator of the form exp((1/4) \sum_{jk} \alpha_{jk} c_j c_k)
  //where \alpha = A\otimes I + B\otimes i\sigma_y
  //A is antisymmetric, B is symmetric
  //so -iA + B is Hermitian, A+iB is anti-Hermitian
  static PassiveFLO from_generator(int qubits, std::vector<double> A, std::vector<double> B){

    std::vector<std::complex<double> > alpha((qubits*(qubits+1))/2, 0.);
    //if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
    for(int i = 0; i < qubits; i++){
      for(int j = i; j < qubits; j++){
	alpha[i + ((j+1)*j)/2] = std::complex<double>(B[dense_fortran(i+1,j+1,qubits)], -A[dense_fortran(i+1,j+1,qubits)]);
      }      
    }
    double tolerance = LAPACK_dlamch("S");
    int M;
    std::vector<double> eigenvals(qubits);
    std::vector<std::complex<double> > eigenvecs(qubits*qubits);
    std::vector<std::complex<double> > work(2*qubits);
    std::vector<double> rwork(7*qubits);
    std::vector<int> iwork(5*qubits);
    std::vector<int> ifail(qubits);
    int info;
    LAPACK_zhpevx("V", //compute eigenvectors & values
		  "A", // all eigenvalues
		  "U", //upper
		  &qubits,
		  &alpha[0],
		  NULL, // VL, we want all so not referenced
		  NULL, // VU, we want all so not referenced
		  NULL, // IL, we want all so not referenced
		  NULL, // IU, we want all so not referenced
		  &tolerance,
		  &M,
		  &eigenvals[0],
		  &eigenvecs[0],
		  &qubits,
		  &work[0],
		  &rwork[0],
		  &iwork[0],
		  &ifail[0],
		  &info);

    std::vector<double> R(2*qubits*2*qubits, 0);
    std::vector<double> alpha2(2*qubits*2*qubits, 0);
    std::cout << std::setw(6) << std::scientific << std::setprecision(5) << std::showpos;
    std::cout << "vals: ";
    for(int i = 0; i < qubits; i++){
      for(int j = 0; j < qubits; j++){
	R[dense_fortran(2*i+1, 2*j+1, 2*qubits)] = eigenvecs[dense_fortran(i+1, j+1, qubits)].real();
	R[dense_fortran(2*i+2, 2*j+2, 2*qubits)] = eigenvecs[dense_fortran(i+1, j+1, qubits)].real();

	R[dense_fortran(2*i+1, 2*j+2, 2*qubits)] = eigenvecs[dense_fortran(i+1, j+1, qubits)].imag();
	R[dense_fortran(2*i+2, 2*j+1, 2*qubits)] =-eigenvecs[dense_fortran(i+1, j+1, qubits)].imag();

	alpha2[dense_fortran(2*i+1, 2*j+1, 2*qubits)] = A[dense_fortran(i+1, j+1, qubits)];
	alpha2[dense_fortran(2*i+2, 2*j+2, 2*qubits)] = A[dense_fortran(i+1, j+1, qubits)];
	
	alpha2[dense_fortran(2*i+1, 2*j+2, 2*qubits)] = B[dense_fortran(i+1, j+1, qubits)];
	alpha2[dense_fortran(2*i+2, 2*j+1, 2*qubits)] = -B[dense_fortran(i+1, j+1, qubits)];
      }
      std::cout << eigenvals[i] << ", ";
    }
    std::cout <<std::endl;
    
    print_fortran(alpha2, 2*qubits);
    std::cout << std::endl;
    print_fortran(matmul_square_double(CblasTrans, CblasNoTrans, R, matmul_square_double(CblasNoTrans, CblasNoTrans, alpha2, R, 2*qubits), 2*qubits), 2*qubits);

    return PassiveFLO(qubits, R);
  }
};

//DecomposedPassive decompose_passive_flo_unitary(std::vector<double> R, int qubits, std::complex<double> phase);
#endif

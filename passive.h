#ifndef PASSIVE_FERMIONIC_LINEAR_OPTICS_H
#define PASSIVE_FERMIONIC_LINEAR_OPTICS_H
#include <vector>
#include <optional>
#include <complex.h>
#include <cmath>
#define lapack_complex_double std::complex<double>
#include <lapack.h>

#include "wrappers.h"

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
  DecomposedPassive decompose();
};

//DecomposedPassive decompose_passive_flo_unitary(std::vector<double> R, int qubits, std::complex<double> phase);
#endif

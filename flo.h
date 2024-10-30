#ifndef FLO_FID_FF_H
#define FLO_FID_FF_H
#define lapack_complex_double std::complex<double>
#include <iostream>
#include <vector>
#include <cblas.h>
#include <complex.h>
#include <cmath>
#include <lapack.h>

#include "pfapack.h"
#include "wrappers.h"
#include "so-factorization.h"

typedef struct DecomposedPassive{
  //R is the orthogonal matrix implementing a passive FLO unitary K such that
  //K exp(sum_j c_{2j}c_{2j+1} lambda_j/2) K^\dagger = U
  //while <0|U|0> = phase
  std::complex<double> phase;
  std::vector<double> l;
  std::vector<double> R;
} DecomposedPassive;

std::complex<double> cb_inner_prod(int qubits, std::vector<int> x, std::vector<double> R, std::complex<double> phase, std::vector<double> l);
DecomposedPassive decompose_passive_flo_unitary(std::vector<double> R, int qubits, std::complex<double> phase);
std::complex<double> inner_prod(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2);
//std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, std::vector<int> y, DecomposedPassive &p, std::vector<double> T, std::vector<double> A);
//std::complex<double> inner_prod_smooth(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2);


std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, int y, DecomposedPassive &p, std::vector<double> A);
#endif

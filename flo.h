#ifndef FERMIONIC_LINEAR_OPTICS_H
#define FERMIONIC_LINEAR_OPTICS_H

#include <iostream>
#include <vector>
#include <cblas.h>
#include <complex.h>
#include <cmath>
#define lapack_complex_double std::complex<double>
#include <lapack.h>

#include "pfapack.h"
#include "wrappers.h"
#include "so-factorization.h"
#include "passive.h"

std::complex<double> cb_inner_prod(int qubits, std::vector<int> x, std::vector<double> R, std::complex<double> phase, std::vector<double> l);
std::complex<double> inner_prod(int qubits, std::vector<double> A1, PassiveFLO K1, std::vector<double> A2, PassiveFLO K2);
//std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, std::vector<int> y, DecomposedPassive &p, std::vector<double> T, std::vector<double> A);
//std::complex<double> inner_prod_smooth(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2);

//we return the inner product, (K1, K1phase), (K2, K2phase), A
//enough information to completely reproduce the KAK decompostion of U
std::tuple<std::complex<double>, PassiveFLO, std::vector<double>, PassiveFLO > aka_to_kak(int qubits, std::vector<double> lambda1, PassiveFLO K,  std::vector<double> lambda2);

std::complex<double> inner_prod_M_P_A(int qubits, std::vector<double> M, DecomposedPassive &p, std::vector<double> A);
std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, int y, DecomposedPassive &p, std::vector<double> A);
#endif

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


std::complex<double> inner_prod(int qubits, std::vector<double> A1, PassiveFLO K1, std::vector<double> A2, PassiveFLO K2);

//we return the inner product, (K1, K1phase), (K2, K2phase), A
//enough information to completely reproduce the KAK decompostion of U
std::tuple<std::complex<double>, PassiveFLO, std::vector<double>, PassiveFLO > aka_to_kak(int qubits, std::vector<double> lambda1, PassiveFLO K,  std::vector<double> lambda2);

std::complex<double> inner_prod_M_P_A(int qubits, std::vector<double> M, DecomposedPassive &p, std::vector<double> A);
std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, int y, DecomposedPassive &p, std::vector<double> A);
std::vector<std::pair<int,int>> reorder_vec(std::vector<int> x);
std::complex<double> anti_passive_vacuum_expectation_value(std::vector<double> lambda);
#endif

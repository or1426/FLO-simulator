#ifndef FLO_SO_FACT_H
#define FLO_SO_FACT_H

#include <iostream>
#include <vector>
#include <cblas.h>
#include <cmath>
#include <lapack.h>
#include "pfapack.h"
#include "wrappers.h"
#include <variant>

std::vector<double> symplectic_orthogonal_factorize(int qubits, std::vector<double> A, std::vector<double> &Q1, std::vector<double> &Q2);

struct SymplecticGivens{
  double c;
  double s;
  int k;
  SymplecticGivens(double a, double b, int k){
    cblas_drotg(&a,&b,&this->c, &this->s);
    //this->c = c;
    //this->s = s;
    this->k = k;
  }
};

struct SymplecticHouseholder{
  std::vector<double> w;
  double tau;
  SymplecticHouseholder(int qubits, double * v_ptr, int k, int v_inc, int part){
    //we construct a symplectic householder which zeros
    //everything below element k of either the even or odd part of v
    this->w = std::vector(2*qubits+1, 0.);
    int effective_length = qubits - k;
    
    cblas_dcopy(effective_length, v_ptr + (2*k+part)*v_inc, 2*v_inc, &w[1+2*k], 2);

    const int TWO = 2;
    LAPACK_dlarfg(&effective_length, &this->w[1+2*k], &this->w[3+2*k], &TWO, &(this->tau));
    this->w[1+2*k] = 1;
  }
};

void apply_left(SymplecticGivens g, std::vector<double> &m, int qubits);

void apply_right(SymplecticGivens g, std::vector<double> &m, int qubits);

void apply_left(SymplecticHouseholder h, std::vector<double> &m, int qubits);

void apply_right(SymplecticHouseholder h, std::vector<double> &m, int qubits);

void apply_left(std::variant<SymplecticGivens, SymplecticHouseholder> op, std::vector<double> &m, int qubits);

void apply_right(std::variant<SymplecticGivens, SymplecticHouseholder> op, std::vector<double> &m, int qubits);

//swap from the symplectic form
//                   [ 0 1]
// \bigoplus_{j=1}^n [-1 0]
//to the one
//[ 0 I]
//[-I 0]
std::vector<double> reshuffled(std::vector<double> A, int n);
#endif

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

struct SymplecticGivens {
  double c;
  double s;
  int k;
  SymplecticGivens(double c, double s, int k){
    this->c = c;
    this->s = s;
    this->k = k;
  }
};

struct SymplecticHouseholder {
  std::vector<double> w;
  SymplecticHouseholder(int qubits, std::vector<double> v, int k){
    double norm2 = 0;
    for(int i = 0; i < qubits; i++){
      norm2 += v[i]*v[i];
    }
    double new_norm = sqrt(norm2 - v[k]*v[k] +(v[k] + sqrt(norm2))*(v[k] + sqrt(norm2)));
    //we pad the w vector with an extra 0 at the front
    //this makes it easy to line up to use lapack to apply the right Householder operations
    this->w = std::vector(2*qubits+1, 0.);
    for(int i = 0; i < qubits; i++){
      w[2*i+1] = v[i]/new_norm;
    }    
    w[2*k+1] += sqrt(norm2)/new_norm;
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

#include "flo-state.h"

#include <complex>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

std::vector<std::complex<double> > random_su2(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform_angle_dist(-M_PI, M_PI);

    double theta = uniform_angle_dist(rng);
    double phase1 = uniform_angle_dist(rng);
    double phase2 = uniform_angle_dist(rng);
    
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    std::vector<std::complex<double> > U(4, std::complex<double> (0.0, 0.0));
    U[dense_fortran(1, 1, 2)] = c * std::exp(std::complex<double>(0.0,  phase1));
    U[dense_fortran(1, 2, 2)] = s * std::exp(std::complex<double>(0.0,  phase2));
    U[dense_fortran(2, 1, 2)] = -s * std::exp(std::complex<double>(0.0, -phase2));
    U[dense_fortran(2, 2, 2)] = c * std::exp(std::complex<double>(0.0, -phase1));
    return U;   
}



std::vector<std::complex<double> > apply_dense_matchgate(
  int n,
  int q,
  std::vector<std::complex<double> >& A,
  std::vector<std::complex<double> >& B,
  std::vector<std::complex<double> >& psi) {
  std::vector<std::complex<double> > out(1<<n, std::complex<double> (0.0, 0.0));
  for(int prefix = 0; prefix < (1<<q); prefix++){    
    for(int suffix = 0; suffix < (1 << (n-q-2)); suffix++){
      const int i00 = prefix | (suffix << (q + 2));


            const int i01 = i00 | (1 << q);
            const int i10 = i00 | (1 << (q + 1));
            const int i11 = i00 | (1 << q) | (1 << (q + 1));

            out[i00] = A[dense_fortran(1,1,2)] * psi[i00] + A[dense_fortran(1,2,2)] * psi[i11];
	    out[i11] = A[dense_fortran(2,1,2)] * psi[i00] + A[dense_fortran(2,2,2)] * psi[i11];
	    out[i01] = B[dense_fortran(2,1,2)] * psi[i10] + B[dense_fortran(2,2,2)] * psi[i01];
            out[i10] = B[dense_fortran(1,1,2)] * psi[i10] + B[dense_fortran(1,2,2)] * psi[i01];
            

    }
  }
  return out;
}

double compare_flo_to_dense(FLOState flo, std::vector<std::complex<double> >& dense) {
    const int n = flo.qubits;

    double max_error = 0.;
    int maximizing_index = 0;
    std::complex<double> flo_val = 0;
    std::complex<double> dense_val = 0;
    double total_error = 0.;
    
    for(int x = 0; x < (1<<n); x++) {
      int popcount = 0;
      for(int q = 0; q < n; q++) {
	popcount += (x >> q) & 1;
      }

      if(popcount % 2 == 0){
	FLOState ketX = FLOState::computational_basis_state(n, x);
	std::complex<double> inner_prod = flo.inner_product(ketX);

	double diff = std::abs(dense[x] - inner_prod);
	total_error += diff;
	if(diff > max_error){
	  max_error = diff;
	  maximizing_index = x;
	}	
      }
    }
    return max_error;
}



int main() {
  int qubits = 6;
  int depth = 30;
  
  std::mt19937_64 rng(1000);
  std::uniform_int_distribution<int> q_dist(0, qubits - 2);

  double max_error = -1;
  for(int test = 0; test < 20; test++){
    FLOState flo(qubits); 
    std::vector<std::complex<double> > dense(1<<qubits);
    dense[0] = 1.;
    
    for(int gate = 0; gate < depth; gate++){
      int qubit = q_dist(rng);
      std::vector<std::complex<double> > A = random_su2(rng);
      std::vector<std::complex<double> > B = random_su2(rng);

      flo.apply_2_qubit_matchgate(qubit, A, B);

      dense = apply_dense_matchgate(qubits, qubit, A, B, dense);      
    }
    double error = compare_flo_to_dense(flo, dense);
    if(error > max_error){
      max_error = error;
    }    
  }
  std::cout << "max error = " << max_error << std::endl;
  return 0;  
}

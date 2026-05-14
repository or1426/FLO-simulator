#include "flo-state.h"

using namespace std::complex_literals;
int main(){
  int qubits = 4;  

  //std::vector<double> lambda = std::vector<double>(qubits/2,0.);
  //lambda[0] = -M_PI/4.;
  //s.apply_antipassive(lambda);

  std::vector<std::complex<double>> A = std::vector<std::complex<double>>(4,0.);
  std::vector<std::complex<double>> B = std::vector<std::complex<double>>(4,0.);

  A[dense_fortran(1,1, 2)] = 1/sqrt(2);
  A[dense_fortran(1,2, 2)] = -1/sqrt(2);
  A[dense_fortran(2,1, 2)] = 1/sqrt(2);
  A[dense_fortran(2,2, 2)] = 1/sqrt(2);

  B[dense_fortran(1,1, 2)] = 1;
  B[dense_fortran(1,2, 2)] = 0;
  B[dense_fortran(2,1, 2)] = 0;
  B[dense_fortran(2,2, 2)] = 1;

  

  for(int i = 6; i < 7; i++){

    std::vector<int> x(qubits);
    int popcount = 0;
    for(int q = 0; q < qubits; q++){
      popcount += (i>>q) & 1;
      x[q] = (i>>q) & 1;
    }
    if(popcount % 2 == 0){
      FLOState s(qubits);
      s.apply_2_qubit_matchgate(1, A, B);

      FLOState ketX = FLOState::computational_basis_state(qubits, x);
      std::complex<double> prod = ketX.inner_product(s);
      std::cout << "final prod x = ";
      for(int q = 0; q < qubits; q++){
	std::cout << x[q];
      }
      std::cout << " " << prod << std::endl;
    }
  }

  
  
  /*
  

    
    

  
  B[dense_fortran(1,1, 2)] = 1;
  B[dense_fortran(2,2, 2)] = 1;
  
  

  std::cout << "R:" << *s.K.phase << std::endl;
  print_fortran(s.K.R, 2*qubits);
  std::cout << std::endl;
  std::cout << "A:" << std::endl;
  for(int i = 0; i < s.A.size(); i++){
    std::cout << s.A[i] << " ";
  }
  std::cout << std::endl;

  if(s.A2.has_value()){
    std::cout << "A2:" << std::endl;
    for(int i = 0; i < s.A2.value().size(); i++){
      std::cout << (s.A2.value())[i] << " ";
    }
    std::cout << std::endl;
  }else{
    std::cout << "no A2" << std::endl;
  }
  std::cout << "s.omega: " << s.omega << std::endl;

  */
  //std::vector<int> x = {1,1,0,0}; 
  //FLOState vacuum = FLOState::computational_basis_state(4,x);;
  //std::complex<double> vacuum_val = s.inner_product(vacuum);

  //FLOState f = FLOState::computational_basis_state(qubits,x);
  
  //std::complex<double> other_val = s.inner_product(f);

  
  
  
  ///std::cout << "0000 inner product: " << vacuum_val << std::endl;


  //std::cout << "0110 inner product: " <<  other_val << std::endl;

  
  
  /*
  std::vector<int> x = {0,0,1,1}; 

  FLOState f = FLOState::computational_basis_state(4,x);
  std::cout << f.omega << " " << f.K.phase.value() << std::endl;
  
  
  print_fortran(f.K.R, 2*4);
  
  DecomposedPassive d = f.K.decompose();
  std::cout << d.phase << std::endl;
  for(double angle : d.l){
    std::cout << angle << ", ";
  }
  std::cout<<std::endl;
  
  std::complex<double> val = cb_inner_prod_adjacent_qubits(4, 2, d, f.A);
  std::cout << val << std::endl;
  */
  return 0;
}

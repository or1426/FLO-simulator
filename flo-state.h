#ifndef FLO_STATE_H
#define FLO_STATE_H
#include <complex.h>
#include <vector>
#include <optional>
#include "wrappers.h"
#include "flo.h"
#include "passive.h"
#include <tuple>
#include <numeric>
#include <stdexcept>

using namespace std::complex_literals;

std::vector<double> KAK_2_by_2_su(std::vector<std::complex<double>> U){
  //Special unitaries have the form
  //[ cos(theta) e^(i phi),   sin(theta) e^{i psi}]
  //[-sin(theta) e^(-i psi), cos(theta) e^{-i phi}]
  //to split into passive and active we want to write this as
  //[e^{i x} 0       ]  [ cos(theta) sin(theta)]  [e^{i y} 0       ]
  //[0       e^{-i x}]  [-sin(theta) cos(theta)]  [0       e^{-i y}]
  //where x+y = phi and x - y = psi
  // => x = (phi+psi)/2, y = (phi-psi)/2
  //we return x,theta,y 
  
  double x = (std::arg(U[dense_fortran(1,1,2)]) + std::arg(U[dense_fortran(1,2,2)]))/2.;
  double y = (std::arg(U[dense_fortran(1,1,2)]) - std::arg(U[dense_fortran(1,2,2)]))/2.;

  double c = (U[dense_fortran(1,1,2)]*std::exp(std::complex<double>(0,-y-x))).real();
  double s = (U[dense_fortran(1,2,2)]*std::exp(std::complex<double>(0,y-x))).real();

  return {x, atan2(s,c), y};
}


/*
  Represent an even parity FLO state in the form
  omega K A |0>
  where omega is a complex number, K is passive and A is anti-passive
  optionally we actually have
  omega A2 K A1 |0>
  and then we convert to the above form if we need to
 */
class FLOState {
 public:
  PassiveFLO K;
  std::vector<double> A;
  std::optional<std::vector<double> > A2;
  std::complex<double> omega;
  int qubits;
  FLOState(int qubits): K(qubits), omega(1.) { 
    //this->K(qubits);// = PassiveFLO(qubits); 
    this->A = std::vector<double>(qubits/2, 0.);
    this->omega = 1;
    this->qubits = qubits;
    this->A2 = std::nullopt;
  }
  
  void apply_antipassive(std::vector<double> lambda){
    if(this->A2){
      for(int i = 0; i < qubits; i++){
	(*(this->A2))[i] += lambda[i];
      }
    }else{
      this->A2 = lambda;
    }
  }

  void apply_passive(CBLAS_TRANSPOSE trans, PassiveFLO K2){
    if(this->A2){
      std::tuple<std::complex<double>, PassiveFLO, std::vector<double>, PassiveFLO> tuple = aka_to_kak(this->qubits, *(this->A2), this->K,  this->A);
      std::cout << "applying passive, complicated case" << std::endl;
      
      //tuple[1]  A(tuple[2])  tuple[3] = A2 K A as orthogonal matrices
      this->K = PassiveFLO::multiply(trans, CblasNoTrans, K2, std::get<3>(tuple));
      this->A = std::get<2>(tuple);
      this->A2 = std::nullopt;
      this->omega *= (*(std::get<1>(tuple).phase));
    }else{
      std::cout << "applying passive, simple case" << std::endl;
      this->K = PassiveFLO::multiply(trans, CblasNoTrans, K2, this->K);
    }
  }

  void apply_passive(PassiveFLO K2){
    this->apply_passive(CblasNoTrans, K2);
  }


  //we require A and B to be special orthogonal matrices
  //in general you could also do the same as this method
  //with unitary matrices for which det(A) = det(B)
  //we apply G(A,B) to qubits i and i+1
  void apply_2_qubit_matchgate(int qubit, std::vector<std::complex<double> >A, std::vector<std::complex<double> >B){
    std::cout << "before"<<std::endl;
    std::cout << "A"<<std::endl;
    print_fortran(A, 2);
    std::cout << "B"<<std::endl;
    print_fortran(B, 2);


    std::vector<double> Adecomp = KAK_2_by_2_su(A);
    std::vector<double> Bdecomp = KAK_2_by_2_su(B);

    std::cout << Adecomp[0] << " " << Adecomp[1] << " " << Adecomp[2] << std::endl;
    std::cout << Bdecomp[0] << " " << Bdecomp[1] << " " << Bdecomp[2] << std::endl;
    
    //we have a passive-active KAK decomposition
    //where the only anti-passive part is the middle of the decomposition of A
    //the R matrix implementing B looks like this
    /*
      cos(t)*cos(x + y),  sin(x + y)*cos(t), -sin(t)*cos(x - y), -sin(t)*sin(x - y)
     -sin(x + y)*cos(t),  cos(t)*cos(x + y),  sin(t)*sin(x - y), -sin(t)*cos(x - y)
      sin(t)*cos(x - y), -sin(t)*sin(x - y),  cos(t)*cos(x + y), -sin(x + y)*cos(t)
      sin(t)*sin(x - y),  sin(t)*cos(x - y),  sin(x + y)*cos(t),  cos(t)*cos(x + y)
     */

    double x = Bdecomp[0], t = Bdecomp[1], y = Bdecomp[2];
    std::vector<double> BRMatrix(4*this->qubits*this->qubits,0.);
    std::vector<double> AK1Matrix(4*this->qubits*this->qubits,0.);
    std::vector<double> AK2Matrix(4*this->qubits*this->qubits,0.);

    for(int i = 0; i < 2*this->qubits; i++){
      BRMatrix[dense_fortran(i+1, i+1, 2*qubits)] = 1;
      AK1Matrix[dense_fortran(i+1, i+1, 2*qubits)] = 1;
      AK2Matrix[dense_fortran(i+1, i+1, 2*qubits)] = 1;
    }
    for(int i = 0; i < 4; i++){
      BRMatrix[dense_fortran(2*qubit+i+1, 2*qubit+i+1,2*this->qubits)] = cos(t)*cos(x + y);
    }

    //top left and bottom right block
    BRMatrix[dense_fortran(2*qubit+2, 2*qubit+1, 2*this->qubits)] = cos(t)*sin(x + y);
    BRMatrix[dense_fortran(2*qubit+1, 2*qubit+2, 2*this->qubits)] =-cos(t)*sin(x + y);
    BRMatrix[dense_fortran(2*qubit+4, 2*qubit+3, 2*this->qubits)] =-cos(t)*sin(x + y);
    BRMatrix[dense_fortran(2*qubit+3, 2*qubit+4, 2*this->qubits)] = cos(t)*sin(x + y);

    //top right and bottom left block
    BRMatrix[dense_fortran(2*qubit+3, 2*qubit+1, 2*this->qubits)] =-sin(t)*cos(x - y);
    BRMatrix[dense_fortran(2*qubit+1, 2*qubit+3, 2*this->qubits)] = sin(t)*cos(x - y);
    
    BRMatrix[dense_fortran(2*qubit+4, 2*qubit+1, 2*this->qubits)] =-sin(t)*sin(x - y);
    BRMatrix[dense_fortran(2*qubit+1, 2*qubit+4, 2*this->qubits)] = sin(t)*sin(x - y);
    
    BRMatrix[dense_fortran(2*qubit+3, 2*qubit+2, 2*this->qubits)] = sin(t)*sin(x - y);
    BRMatrix[dense_fortran(2*qubit+2, 2*qubit+3, 2*this->qubits)] =-sin(t)*sin(x - y);
    
    BRMatrix[dense_fortran(2*qubit+4, 2*qubit+2, 2*this->qubits)] =-sin(t)*cos(x - y);
    BRMatrix[dense_fortran(2*qubit+2, 2*qubit+4, 2*this->qubits)] = sin(t)*cos(x - y);
    
    //the two "K-type" parts of A give us terms like
    /*
      cos(x), sin(x), 0, 0
     -sin(x), cos(x), 0, 0
      0, 0,  cos(x), sin(x)
      0, 0, -sin(x), cos(x)
    */
    //and they have a phase K|0> = e^{ix}|0>

    for(int i = 0; i < 4; i++){
      AK1Matrix[dense_fortran(2*qubit+i+1,2*qubit+i+1, 2*this->qubits)] = cos(Adecomp[0]);
      AK2Matrix[dense_fortran(2*qubit+i+1,2*qubit+i+1, 2*this->qubits)] = cos(Adecomp[2]);
    }

    AK1Matrix[dense_fortran(2*qubit+2,2*qubit+1, 2*this->qubits)] =-sin(Adecomp[0]);
    AK1Matrix[dense_fortran(2*qubit+4,2*qubit+3, 2*this->qubits)] =-sin(Adecomp[0]);
    AK1Matrix[dense_fortran(2*qubit+1,2*qubit+2, 2*this->qubits)] = sin(Adecomp[0]);
    AK1Matrix[dense_fortran(2*qubit+3,2*qubit+4, 2*this->qubits)] = sin(Adecomp[0]);

    AK2Matrix[dense_fortran(2*qubit+2,2*qubit+1, 2*this->qubits)] =-sin(Adecomp[2]);
    AK2Matrix[dense_fortran(2*qubit+4,2*qubit+3, 2*this->qubits)] =-sin(Adecomp[2]);
    AK2Matrix[dense_fortran(2*qubit+1,2*qubit+2, 2*this->qubits)] = sin(Adecomp[2]);
    AK2Matrix[dense_fortran(2*qubit+3,2*qubit+4, 2*this->qubits)] = sin(Adecomp[2]);


    for(int i = 0; i < AK1Matrix.size(); i++){
      if(abs(AK1Matrix[i]) < 1e-10){
	AK1Matrix[i] = 0;
      }
      if(abs(AK2Matrix[i]) < 1e-10){
	AK2Matrix[i] = 0;
      }
    }
    
    std::cout << "AK1" <<std::endl;
    print_fortran(AK1Matrix, 2*this->qubits);
    std::cout << std::endl;

    std::cout << "AK2" <<std::endl;
    print_fortran(AK2Matrix, 2*this->qubits);
    std::cout << std::endl;

    std::cout << "BR" <<std::endl;
    print_fortran(BRMatrix, 2*this->qubits);
    std::cout << std::endl;


    std::vector<double> antipassive_vector(this->qubits/2, 0.);
    std::cout << "qubit/2 = " << qubit/2 << std::endl;
    
    antipassive_vector[qubit/2] = Adecomp[1]; //note that if qubit is odd then directly applying this antipassive will be wrong
    std::cout << "applying BR matrix" << std::endl;
    this->apply_passive(PassiveFLO(this->qubits, BRMatrix, std::complex<double>(1.,0.)));
    std::cout << "applying AK2 matrix" << std::endl;
    this->apply_passive(PassiveFLO(this->qubits, AK2Matrix, std::exp(std::complex<double>(0, Adecomp[2]))));
    if(qubit % 2 == 0){
      this->apply_antipassive(antipassive_vector);
    }else{
      //the qubit we're applying the antipassive to is wrong, it should be one higher
      //this is because out antipassives are defined as acting on qubits (2*i, 2*i+1)
      //but we want to apply one to (2+i+1, 2*i+2)

      PassiveFLO permutation(this->qubits);

      for(int i = 0; i < 6; i++){
	permutation.R[dense_fortran(2*(qubit - 1) + 1 + i, 2*(qubit - 1) + 1 + i, 2*this->qubits)] = 0;
	permutation.R[dense_fortran(2*(qubit - 1) + 1 + i, 2*(qubit - 1) + 1 + ((i+2) % 6), 2*this->qubits)] = 1;	
      }
      std::cout << "permutation = " << std::endl;
      print_fortran(permutation.R, 2*qubits);
      std::cout << std::endl;
      DecomposedPassive d = permutation.decompose();
      std::cout << d.phase << std::endl;


      this->apply_passive(CblasNoTrans, permutation);
      this->apply_antipassive(antipassive_vector);
      this->apply_passive(CblasTrans, permutation);
    }
    std::cout << "applying AK1 matrix" << std::endl;
    this->apply_passive(PassiveFLO(this->qubits, AK1Matrix, std::exp(std::complex<double>(0., Adecomp[0]))));
  }

  std::complex<double> inner_product(FLOState &other){
    if(this->A2){
      PassiveFLO identity(this->qubits);
      identity.phase = std::complex<double>(1.,0.);
      std::cout << identity.phase.value() << std::endl;
      this->apply_passive(identity);
    }
    if(other.A2){
      PassiveFLO identity(this->qubits);
      identity.phase = std::complex<double>(1.,0.);
      std::cout << identity.phase.value() << std::endl;
      other.apply_passive(identity);
    }
    std::cout << "this->K.phase = " << this->K.phase.value() << " this->omega " << this->omega << std::endl;
    std::cout << "other.K.phase = " << other.K.phase.value() << " other.omega " << other.omega << std::endl;

    for(int i = 0; i < this->qubits/2; i++){
      std::cout << this->A[i] << " " << other.A[i] << ", ";
    }
    std::cout << std::endl;
    print_fortran(this->K.R, 2*this->qubits);
    print_fortran(other.K.R, 2*other.qubits);
    std::cout << std::endl;
    return inner_prod(this->qubits, other.A, other.K, this->A, this->K)*this->omega*std::conj(other.omega);			       
  }

  static FLOState computational_basis_state(int qubits, std::vector<int> x){

    int popcount = std::accumulate(x.begin(), x.end(), 0);
    if((popcount % 2) != 0){
      //we only ever work in the even parity subspace
      throw std::invalid_argument("We can only express computational basis states with even parity");
    }
    
    FLOState state(qubits);
    //we make a FLO state with the first sum(x)/2 pairs of qubits set to |11>
    //then add a passive flo permutation to move these to the right places

    for(int i = 0; i < popcount/2; i++){      
      state.A[i] =  M_PI/2.;
    }

    //this permutation swaps the elements of x so all the non-zero entries are at the start
    std::vector<std::pair<int,int> > permutation = reorder_vec(x);
    std::cout << "generating cb state" << std::endl;
    
     for(const std::pair<int,int> &pair: permutation){
       for(int i = 0; i < 2*qubits;i++){
	 std::swap(state.K.R[dense_fortran(i+1, 2*pair.first+1,  2*qubits)],
		   state.K.R[dense_fortran(i+1, 2*pair.second+1, 2*qubits)]);
	 std::swap(state.K.R[dense_fortran(i+1, 2*pair.first+2,  2*qubits)],
		   state.K.R[dense_fortran(i+1, 2*pair.second+2, 2*qubits)]);
       }
     }

     DecomposedPassive d = state.K.decompose();
     std::cout << d.phase << std::endl;
    
    return state;
  }
  
};

#endif

#include "flo.h"
#include "passive.h"
#include "json.hpp"
#include <iomanip>
using json = nlohmann::json;

using namespace std::complex_literals;


double decompose_passive_test(json j){  
  int qubits = j["qubits"];
  std::vector<double> R = j["R"];
  double phaseR = j["phaseReal"];
  double phaseI = j["phaseImag"];
  std::complex<double> phase = phaseR + 1.i * phaseI;
  
  DecomposedPassive p = PassiveFLO(qubits, R, phase).decompose();

  double max_error = abs(p.phase - phase);
  
  matrix_conjugate_inplace_double(R, p.R, 2*qubits);
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      if(std::abs(R[dense_fortran(i+1, j+1, 2*qubits)]) < 1e-13){
	R[dense_fortran(i+1, j+1, 2*qubits)] = 0;
      }
      if(std::abs(R[dense_fortran(i+1, j+1, 2*qubits)]-1) < 1e-13){
	R[dense_fortran(i+1, j+1, 2*qubits)] = 1;
      }      
    }
  }
  
  for(int i = 0; i < qubits; i++){
    double c = std::cos(p.l[i]);
    double s = -std::sin(p.l[i]);

    R[dense_fortran(2*i+1, 2*i+1, 2*qubits)] -= c;
    R[dense_fortran(2*i+2, 2*i+2, 2*qubits)] -= c;
    R[dense_fortran(2*i+1, 2*i+2, 2*qubits)] -= s;
    R[dense_fortran(2*i+2, 2*i+1, 2*qubits)] += s;
  }

  for(auto it = R.begin(); it != R.end(); it++){
    if(abs(*it) > max_error){
      max_error = abs(*it);
    }
  }
  return max_error;
}

double inner_product_test(json j){
  int qubits = j["qubits"];
  std::vector<double> R1 = j["R1"];
  std::vector<double> R2 = j["R2"];
  double prodR = j["prodR"];
  double prodI = j["prodI"];
  // <0| A1^dagger K_1^\dagger K_2 A2 |0> 
  std::complex<double> correct_prod = prodR + 1.i * prodI;
  
  double phase1R = j["phase1R"];
  double phase1I = j["phase1I"];
  double phase2R = j["phase2R"];
  double phase2I = j["phase2I"];
  std::complex<double> phase1 = phase1R + phase1I*1.i;
  std::complex<double> phase2 = phase2R + phase2I*1.i;
  std::vector<double> A1 = j["l1"];
  std::vector<double> A2 = j["l2"];
  PassiveFLO K1(qubits, R1, phase1);
  PassiveFLO K2(qubits, R2, phase2);
  std::complex<double> prod = inner_prod(qubits, A1, K1, A2, K2);

  return abs(prod - correct_prod);
}


double cb_inner_prod_adjacent_qubits_test(json j){
  int qubits = j["qubits"];
  std::vector<double> R = j["R"];
  std::vector<int> c_vec = j["c_vec"];
  double pythonvalR = j["pythonvalReal"];
  double pythonvalI = j["pythonvalImag"];
  std::complex<double> pythonval = pythonvalR + 1.i*pythonvalI;
  double phaseR = j["phaseReal"];
  double phaseI = j["phaseImag"];
  std::complex<double> phase = phaseR + 1.i * phaseI;
  std::vector<double> l = j["l"];

  int y = 0;
  for(int k = 0; k < qubits/2; k++){
    if(c_vec[4*k] == 1){
      y ^= (1<<k);
    }
  }	
  DecomposedPassive p = PassiveFLO(qubits, R, phase).decompose();

  std::complex<double> val = cb_inner_prod_adjacent_qubits(qubits, y, p,  l);

  return abs(val - pythonval);  
}

double symplectic_orthogonal_factorize_test(json j){
  int qubits = j["qubits"];
  std::vector<double> R = j["R"];

  std::vector<double> Q1(4*qubits*qubits);
  std::vector<double> Q2(4*qubits*qubits);

  std::vector<double> T = symplectic_orthogonal_factorize(qubits, R, Q1, Q2);
  //things to test:
  // 1. Q1 & Q2 are orthogonal
  // 2. Q1 & Q2 are symplectic
  // Q1 R Q2 = T
  // T has the correct structure
  std::vector<double> id1 = matmul_square_double(CblasNoTrans, CblasTrans, Q1, Q1, 2*qubits);
  std::vector<double> id2 = matmul_square_double(CblasNoTrans, CblasTrans, Q2, Q2, 2*qubits);

  std::vector<double> J(4*qubits*qubits, 0.);//symplectic form
  for(int i = 0; i < qubits; i++){
    J[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = 1;
    J[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -1;
  }

  std::vector<double> J1 = matmul_square_double(CblasNoTrans, CblasNoTrans, Q1, J, 2*qubits);
  J1 = matmul_square_double(CblasNoTrans, CblasTrans, J1, Q1, 2*qubits);
  
  std::vector<double> J2 = matmul_square_double(CblasNoTrans, CblasNoTrans, Q2, J, 2*qubits);
  J2 = matmul_square_double(CblasNoTrans, CblasTrans, J2, Q2, 2*qubits);

  std::vector<double> T2 = matmul_square_double(CblasNoTrans, CblasNoTrans, Q1, R, 2*qubits);
  T2 = matmul_square_double(CblasNoTrans, CblasNoTrans, T2, Q2, 2*qubits);

  double max_error = -1;
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      //test both matrices which should be identity matrices
      if(i == j){
	max_error = std::max(max_error, abs(id1[dense_fortran(i+1, j+1, 2*qubits)] - 1));
	max_error = std::max(max_error, abs(id2[dense_fortran(i+1, j+1, 2*qubits)] - 1));
      }else{
	max_error = std::max(max_error, abs(id1[dense_fortran(i+1, j+1, 2*qubits)]));
	max_error = std::max(max_error, abs(id2[dense_fortran(i+1, j+1, 2*qubits)]));
      }
      
      //test both matrices which should be symplectic forms
      max_error = std::max(max_error, abs(J1[dense_fortran(i+1, j+1, 2*qubits)] - J[dense_fortran(i+1, j+1, 2*qubits)]));
      max_error = std::max(max_error, abs(J2[dense_fortran(i+1, j+1, 2*qubits)] - J[dense_fortran(i+1, j+1, 2*qubits)]));
      
      //test that the factorization closely reproduces the original matrix
      max_error = std::max(max_error, abs(T2[dense_fortran(i+1, j+1, 2*qubits)] - T[dense_fortran(i+1, j+1, 2*qubits)]));
      
    }
  }
  //test that T has the right form
  //T = reshuffled(T,qubits);
  for(int i = 0; i < qubits; i++){
    for(int j = 0; j < qubits; j++){
      //test top left block is identity
      if(i == j){
	max_error = std::max(max_error, abs(T[dense_fortran(2*i+1, 2*j+1, 2*qubits)] - 1));
      }else{
	max_error = std::max(max_error, abs(T[dense_fortran(2*i+1, 2*j+1, 2*qubits)]));
      }
      //test top right and bottom left blocks are 0
      max_error = std::max(max_error, abs(T[dense_fortran(2*i+1, 2*j+2, 2*qubits)]));
      max_error = std::max(max_error, abs(T[dense_fortran(2*i+2, 2*j+1, 2*qubits)]));
      
      //test bottom right block is direct sum of 2x2 blocks
      if(abs(i-j) > 1){
	max_error = std::max(max_error, abs(T[dense_fortran(2*i+2, 2*j+2, 2*qubits)]));
      }  
    }
  }
  
  return max_error;
}


double aka_to_kak_test(json j){
  int qubits = j["qubits"];
  std::vector<double> lambda1 = j["lambda1"];
  std::vector<double> lambda2 = j["lambda2"];
  std::vector<double> R = j["R"];
  double phaseR = j["phaseReal"];
  double phaseI = j["phaseImag"];
  PassiveFLO K(qubits, R, std::complex<double>(phaseR, phaseI));
  std::tuple<std::complex<double>, PassiveFLO, std::vector<double>, PassiveFLO > tuple = aka_to_kak(qubits, lambda1, K, lambda2);

  std::complex<double> val = std::get<0>(tuple);
  std::complex<double> pythonVal(j["pythonValR"], j["pythonValI"]);

  double error = std::min(abs(val - pythonVal), abs(val + pythonVal));
  std::vector<double> lambda = std::get<2>(tuple);
  std::vector<double> A_R_matrix(2*qubits*2*qubits, 0.);
  std::vector<double> A1_R_matrix(2*qubits*2*qubits, 0.);
  std::vector<double> A2_R_matrix(2*qubits*2*qubits, 0.);
  
  for(int i = 0; i < qubits/2; i++){
    //A
    A_R_matrix[dense_fortran(4*i+1, 4*i+1, 2*qubits)] = cos(-lambda[i]);
    A_R_matrix[dense_fortran(4*i+2, 4*i+2, 2*qubits)] = cos(-lambda[i]);
    A_R_matrix[dense_fortran(4*i+3, 4*i+3, 2*qubits)] = cos(-lambda[i]);
    A_R_matrix[dense_fortran(4*i+4, 4*i+4, 2*qubits)] = cos(-lambda[i]);

    A_R_matrix[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = sin(-lambda[i]);
    A_R_matrix[dense_fortran(4*i+2, 4*i+4, 2*qubits)] = -sin(-lambda[i]);
    A_R_matrix[dense_fortran(4*i+3, 4*i+1, 2*qubits)] = -sin(-lambda[i]);
    A_R_matrix[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = sin(-lambda[i]);
    //A1
    A1_R_matrix[dense_fortran(4*i+1, 4*i+1, 2*qubits)] = cos(-lambda1[i]);
    A1_R_matrix[dense_fortran(4*i+2, 4*i+2, 2*qubits)] = cos(-lambda1[i]);
    A1_R_matrix[dense_fortran(4*i+3, 4*i+3, 2*qubits)] = cos(-lambda1[i]);
    A1_R_matrix[dense_fortran(4*i+4, 4*i+4, 2*qubits)] = cos(-lambda1[i]);

    A1_R_matrix[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = sin(-lambda1[i]);
    A1_R_matrix[dense_fortran(4*i+2, 4*i+4, 2*qubits)] = -sin(-lambda1[i]);
    A1_R_matrix[dense_fortran(4*i+3, 4*i+1, 2*qubits)] = -sin(-lambda1[i]);
    A1_R_matrix[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = sin(-lambda1[i]);
    //A2
    A2_R_matrix[dense_fortran(4*i+1, 4*i+1, 2*qubits)] = cos(-lambda2[i]);
    A2_R_matrix[dense_fortran(4*i+2, 4*i+2, 2*qubits)] = cos(-lambda2[i]);
    A2_R_matrix[dense_fortran(4*i+3, 4*i+3, 2*qubits)] = cos(-lambda2[i]);
    A2_R_matrix[dense_fortran(4*i+4, 4*i+4, 2*qubits)] = cos(-lambda2[i]);

    A2_R_matrix[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = sin(-lambda2[i]);
    A2_R_matrix[dense_fortran(4*i+2, 4*i+4, 2*qubits)] = -sin(-lambda2[i]);
    A2_R_matrix[dense_fortran(4*i+3, 4*i+1, 2*qubits)] = -sin(-lambda2[i]);
    A2_R_matrix[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = sin(-lambda2[i]);    
  }
  //should be R^T R = I
  std::vector<double> R1 = matmul_square_double(A2_R_matrix, matmul_square_double(R, A1_R_matrix, 2*qubits), 2*qubits);
  std::vector<double> R2 = matmul_square_double(std::get<1>(tuple).R, matmul_square_double(A_R_matrix, std::get<3>(tuple).R, 2*qubits), 2*qubits);
  
  std::vector<double> identity = matmul_square_double(CblasTrans, CblasNoTrans, R1, R2, 2*qubits);

  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      if(i == j){
	error = std::max(error, abs(identity[dense_fortran(i+1, j+1, 2*qubits)] - 1));
      }else{
	error = std::max(error, abs(identity[dense_fortran(i+1, j+1, 2*qubits)]));
      }
    }
  }

  
  return error;  
}

double MKA_test(json j){
  int qubits = j["qubits"];
  std::vector<double> M = j["M"];
  std::vector<double> R = j["R"];
  std::complex<double> Rphase(j["RphaseR"], j["RphaseI"]);
  std::vector<double> A = j["A"];

  std::complex<double> pythonProd(j["prodR"], j["prodI"]);
  
  DecomposedPassive p = PassiveFLO(qubits, R, Rphase).decompose();
  
  std::complex<double> prod = inner_prod_M_P_A(qubits, M, p, A);
  //std::cout << prod << ", " << pythonProd << std::endl;
  return abs(prod - pythonProd);
  
}

int main(int argc, char * argv[])
{
  json j;
  int count = 0;

  double max_error_decompose_passive = -1;
  double max_error_flo_ip = -1;
  double max_error_cb_ip = -1;
  double max_error_so = -1;
  double max_error_aka_kak = -1;
  double max_error_MKA = -1;
  int decompose_passive_count = 0;
  int flo_ip_count = 0;
  int cb_ip_count = 0;
  int so_count = 0;
  int aka_kak_count = 0;
  int mka_prod_count = 0;
  while(true){
    //this try/except thing seems suboptimal/ugly
    //I think this json library really doesn't expect to be dealing with streams of json objects
    try{
      std::cin >> j;

      if(j["type"] == std::string("passive-decomp")){
	decompose_passive_count += 1;
	double val = decompose_passive_test(j);
	if(val > max_error_decompose_passive){
	  max_error_decompose_passive = val;
	}
      }else if(j["type"] == std::string("flo-inner-product")){
	flo_ip_count += 1;
	double val = inner_product_test(j);
	if(val > max_error_flo_ip){
	  max_error_flo_ip = val;
	}
      }else if(j["type"] == std::string("cb-inner-product")){
	cb_ip_count += 1;
	double val = cb_inner_prod_adjacent_qubits_test(j);
	if(val > max_error_cb_ip){
	  max_error_cb_ip = val;
	}
      }else if(j["type"] == std::string("symplectic-orthogonal-decomposition")){
	so_count += 1;
	double val = symplectic_orthogonal_factorize_test(j);
	if(val > max_error_so){
	  max_error_so = val;
	}	 
	}else if(j["type"] == std::string("aka_kak")){
	aka_kak_count += 1;
	double val = aka_to_kak_test(j);
	if(val > max_error_aka_kak){
	  max_error_aka_kak = val;
	}
      }else if(j["type"] == std::string("mka_prod")){
	mka_prod_count += 1;
	double val = MKA_test(j);
	if(val > max_error_MKA){
	  max_error_MKA = val;
	}	
      }      
      count += 1;
    } catch(nlohmann::json::parse_error e) {
      break;
    }
  }
  std::cout << std::left;
  std::cout << std::setw(20) << "type " << std::setw(20) << "max error " << "count" << std::endl;
  std::cout << std::setw(20) << "decompose passive " << std::setw(20) << max_error_decompose_passive <<decompose_passive_count << std::endl;
  std::cout << std::setw(20) << "cb inner product " << std::setw(20) << max_error_cb_ip << cb_ip_count << std::endl;
  std::cout << std::setw(20) << "flo inner product " << std::setw(20) << max_error_flo_ip << flo_ip_count << std::endl;
  std::cout << std::setw(20) << "so decomposition " << std::setw(20) << max_error_so << so_count << std::endl;
  std::cout << std::setw(20) << "aka to kak form " << std::setw(20) << max_error_aka_kak << aka_kak_count << std::endl;
  std::cout << std::setw(20) << "MKA " << std::setw(20) << max_error_MKA << mka_prod_count << std::endl;
  
  /*
  std::string helpmessage("args:\n\t-d for testing the decomposition of passive FLO unitaries\n\t-c for computational basis inner products\n\t-i for general inner products");

  if(argc == 2){
    std::string arg = std::string(argv[1]);
    
    if(arg == "--decompose" || arg == "-d"){
      decompose_passive_test();
    }
    else if(arg == "--computational-basis" || arg == "-c"){
      cb_inner_prod_adjacent_qubits_test();
    }
    else if(arg == "--inner-product" || arg == "-i"){
      inner_product_test();
    }
    else{
      std::cout <<  helpmessage << std::endl;
    }
  }else{
    std::cout <<  helpmessage << std::endl;
  }
  */
  return 0;
}

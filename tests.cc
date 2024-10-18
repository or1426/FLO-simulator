#include "ff.h"
#include "json.hpp"
#include <iomanip>
using json = nlohmann::json;

using namespace std::complex_literals;


int decompose_passive_test(){
  json j;
  int count = 0;
  std::cin >> count;

  double max_error = -1;
  for(int idx = 0; idx < count; idx++){
    std::cin >>j;
    int qubits = j["qubits"];
    std::vector<double> R = j["R"];
    double phaseR = j["phaseReal"];
    double phaseI = j["phaseImag"];
    std::complex<double> phase = phaseR + 1.i * phaseI;

    DecomposedPassive p = decompose_passive_flo_unitary(R, qubits, phase);
    if(abs(p.phase - phase) > max_error){
      max_error = abs(p.phase - phase);
      std::cout << "found phase " << p.phase << " python sent "<< phase << std::endl;
    }
    
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
      double s = std::sin(p.l[i]);
      std::vector<double> errors(4);
      
      errors[0] = abs(c - R[dense_fortran(2*i+1, 2*i+1, 2*qubits)]);
      errors[1] = abs(c - R[dense_fortran(2*i+2, 2*i+2, 2*qubits)]);
      errors[2] = abs(s - R[dense_fortran(2*i+1, 2*i+2, 2*qubits)]);
      errors[3] = abs(s + R[dense_fortran(2*i+2, 2*i+1, 2*qubits)]);
      for(const double &error : errors){      
	if(error > max_error){
	  max_error = error;
	}
      }
    }
    
  }
  std::cout << "max error: " << max_error << std::endl;
  return 0;
}

int inner_product_test(){
  json j;
  int count = 0;
  std::cin >> count;

  double max_error = -1;
  for(int idx = 0; idx < count; idx++){
    std::cin >>j;
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
    
    std::complex<double> prod = inner_prod(qubits, A1, R1, phase1, A2, R2, phase2);
    if(abs(prod - correct_prod) > max_error){
      max_error = abs(prod - correct_prod);
      std::cout << std::setprecision (10) << "true value: " << correct_prod << ", c++ value: "  << prod <<std::endl;
    }
    
  }
  std::cout << count << " tests, max error: " << max_error << std::endl;
  return 0;
}


int cb_inner_prod_adjacent_qubits_test(){
  json j;
  int count = 0;
  std::cin >> count;
  std::cout << "count = " << count << std::endl;
  double max_error = -1;
  for(int i = 0; i < count; i++){
    std::cin >>j;
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
    //std::cout << "y = ";
    for(int k = 0; k < qubits/2; k++){
      if(c_vec[4*k] == 1){
	y ^= (1<<k);
      }
    }	
    DecomposedPassive p = decompose_passive_flo_unitary(R, qubits, phase);

    std::complex<double> val = cb_inner_prod_adjacent_qubits(qubits, y, p,  l);
    if(abs(val - pythonval) > max_error){
      max_error = abs(val - pythonval);
      std::cout << "python: " << pythonval << " c++: " << val << std::endl;
    }
  }
  
  std::cout << "max error: " << max_error << std::endl;

  return 0;
}



int main(int argc, char * argv[])
{
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
  return 0;
}

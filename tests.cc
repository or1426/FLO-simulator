#include "ff.h"
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
  
  DecomposedPassive p = decompose_passive_flo_unitary(R, qubits, phase);
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
  
  std::complex<double> prod = inner_prod(qubits, A1, R1, phase1, A2, R2, phase2);
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
  //std::cout << "y = ";
  for(int k = 0; k < qubits/2; k++){
    if(c_vec[4*k] == 1){
      y ^= (1<<k);
    }
  }	
  DecomposedPassive p = decompose_passive_flo_unitary(R, qubits, phase);
  
  std::complex<double> val = cb_inner_prod_adjacent_qubits(qubits, y, p,  l);
  return abs(val - pythonval);  
}



int main(int argc, char * argv[])
{
  json j;
  int count = 0;

  double max_error_decompose_passive = -1;
  double max_error_flo_ip = -1;
  double max_error_cb_ip = -1;

  int decompose_passive_count = 0;
  int flo_ip_count = 0;
  int cb_ip_count = 0;
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

#include "wrappers.h"
#include "passive.h"
#include <vector>

int main() {

  std::vector<double> A(4, 0);
  std::vector<double> B(4, 0);

  A[dense_fortran(1,2,2)] = 1.7;
  A[dense_fortran(2,1,2)] = -1.7;

  B[dense_fortran(1,1,2)] = 2.2;
  B[dense_fortran(1,2,2)] = M_PI;
  B[dense_fortran(2,1,2)] = M_PI;
  B[dense_fortran(2,2,2)] = 5;
  
  PassiveFLO p = PassiveFLO::from_generator(2, A, B);

}

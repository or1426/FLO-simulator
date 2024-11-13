#include "so-factorization.h"
#include <iomanip>


void apply_left(SymplecticGivens g, std::vector<double> &m, int qubits){
  int n = 2*qubits;
  cblas_drot(n, &m[dense_fortran(2*g.k+1, 1, n)], n, &m[dense_fortran(2*g.k+2, 1, n)], n, g.c, g.s);
}

void apply_right(SymplecticGivens g, std::vector<double> &m, int qubits){
  int n = 2*qubits;
  int inc = 1;
  double minus_s = -g.s;
  cblas_drot(n, &m[dense_fortran(1, 2*g.k+1, n)], inc, &m[dense_fortran(1, 2*g.k+2, n)], inc, g.c, -g.s);
}

void apply_left(SymplecticHouseholder h, std::vector<double> &m, int qubits){
  //a symplectic Householder operation is essentially two normal Householder operations
  std::vector<double> work(2*qubits);
  int n = 2*qubits;
  int inc = 1;

  LAPACK_dlarf("L",
	       &n, // number of rows
	       &n, //cols
	       &h.w[1],
	       &inc, //incv		
	       &h.tau, //tau	
	       &m[0], 
	       &n,
	       &work[0]);	
  LAPACK_dlarf("L",
	       &n, // number of rows
	       &n, //cols
	       &h.w[0],
	       &inc, //incv
	       &h.tau, //tau
	       &m[0], 
	       &n,
	       &work[0]);  
}

void apply_right(SymplecticHouseholder h, std::vector<double> &m, int qubits){
  //a symplectic Householder operation is essentially two normal Householder operations
  std::vector<double> work(2*qubits);
  int n = 2*qubits;
  int inc = 1;

  LAPACK_dlarf("R",
	       &n, // number of rows
	       &n, //cols
	       &h.w[1],
	       &inc, 
	       &h.tau, 
	       &m[0], 
	       &n,
	       &work[0]);	
  LAPACK_dlarf("R",
	       &n, // number of rows
	       &n, //cols
	       &h.w[0],
	       &inc, 
	       &h.tau, 
	       &m[0], 
	       &n,
	       &work[0]);
}

void apply_left(std::variant<SymplecticGivens, SymplecticHouseholder> op, std::vector<double> &m, int qubits){
  if(std::holds_alternative<SymplecticGivens>(op)){
    apply_left(std::get<SymplecticGivens>(op), m, qubits);
  }
  if(std::holds_alternative<SymplecticHouseholder>(op)){
    apply_left(std::get<SymplecticHouseholder>(op), m, qubits);
  }
}

void apply_right(std::variant<SymplecticGivens, SymplecticHouseholder> op, std::vector<double> &m, int qubits){
  if(std::holds_alternative<SymplecticGivens>(op)){
    apply_right(std::get<SymplecticGivens>(op), m, qubits);
  }
  if(std::holds_alternative<SymplecticHouseholder>(op)){
    apply_right(std::get<SymplecticHouseholder>(op), m, qubits);
  }
}

std::vector<double> reshuffled(std::vector<double> A, int n){
  //swap from the symplectic form
  //                   [ 0 1]
  // \bigoplus_{j=1}^n [-1 0]
  //to the one
  //[ 0 I]
  //[-I 0]
  //is there a smarter way to do this?
  std::vector<double> B(4*n*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      B[dense_fortran(i+1, j+1, 2*n)] = A[dense_fortran(2*i+1, 2*j+1, 2*n)];
      B[dense_fortran(n+i+1, j+1, 2*n)] = A[dense_fortran(2*i+2, 2*j+1, 2*n)];
      B[dense_fortran(i+1, n+j+1, 2*n)] = A[dense_fortran(2*i+1, 2*j+2, 2*n)];
      B[dense_fortran(n+i+1, n+j+1, 2*n)] = A[dense_fortran(2*i+2, 2*j+2, 2*n)];
    }
  }
  return B;
}


std::vector<double> reshuffled_inverse(std::vector<double> A, int n){
  //swap from the symplectic form
  //[ 0 I]
  //[-I 0]
  //to the one
  //                   [ 0 1]
  // \bigoplus_{j=1}^n [-1 0]
  std::vector<double> B(4*n*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      B[dense_fortran(2*i+1, 2*j+1, 2*n)] = A[dense_fortran(i+1, j+1, 2*n)];
      B[dense_fortran(2*i+2, 2*j+1, 2*n)] = A[dense_fortran(n+i+1, j+1, 2*n)];
      B[dense_fortran(2*i+1, 2*j+2, 2*n)] = A[dense_fortran(i+1, n+j+1, 2*n)];
      B[dense_fortran(2*i+2, 2*j+2, 2*n)] = A[dense_fortran(n+i+1, n+j+1, 2*n)];
    }
  }
  return B;
}


int selectfn(const double * r, const double *i){
  return abs(*i) < 1e-13;
}

std::vector<double> symplectic_orthogonal_factorize(int qubits, std::vector<double> A, std::vector<double> & Q1, std::vector<double> &Q2){

  std::vector<double> v(qubits); //scratch space vector used for computing Householder reflections
  std::vector<std::variant<SymplecticGivens, SymplecticHouseholder> > left_ops;
  std::vector<std::variant<SymplecticGivens, SymplecticHouseholder> > right_ops;
  for(int k = 0; k < qubits-1; k++){
    //eliminate below diagonal of lower left block
    left_ops.push_back(SymplecticHouseholder(qubits, &A[dense_fortran(1, 2*k+1,2*qubits)], k, 1, 1));
    apply_left(left_ops.back(), A, qubits);        
    
    //eliminate diagonal of lower left block    
    left_ops.push_back(SymplecticGivens(A[dense_fortran(2*k+1, 2*k+1,2*qubits)], A[dense_fortran(2*k+2, 2*k+1,2*qubits)], k));
    apply_left(left_ops.back(), A, qubits);
            
    //eliminate below diagonal of upper left block
    left_ops.push_back(SymplecticHouseholder(qubits, &A[dense_fortran(1, 2*k+1,2*qubits)], k, 1,0));
    apply_left(left_ops.back(), A, qubits);
    
    //eliminate (from the right) right of diagonal of lower left block
    if(k+1 < qubits - 1){
      right_ops.push_back(SymplecticHouseholder(qubits, &A[dense_fortran(2*k+2, 1, 2*qubits)], k+1, 2*qubits, 0));
      apply_right(right_ops.back(), A, qubits);      
    }
    
    //eliminate the remaining above diagonal of lower left block
    right_ops.push_back(SymplecticGivens(A[dense_fortran(2*k+2, 2*(k+1)+2,2*qubits)], A[dense_fortran(2*k+2, 2*(k+1)+1,2*qubits)], k+1));
    apply_right(right_ops.back(), A, qubits);
    
    //eliminate (from right) right of diagonal in lower right block
    if(k+1 < qubits - 1){
      right_ops.push_back(SymplecticHouseholder(qubits, &A[dense_fortran(2*k+2, 1, 2*qubits)], k+1, 2*qubits, 1));      
      apply_right(right_ops.back(), A, qubits);

    }

  }
  //finally delete the lower right element of the lower left block
  left_ops.push_back(SymplecticGivens(A[dense_fortran(2*qubits - 2 + 1, 2*qubits - 2 + 1,2*qubits)],
				      A[dense_fortran(2*qubits-1+1, 2*qubits - 2 + 1,2*qubits)], qubits - 1));
  apply_left(left_ops.back(), A, qubits);
  
  std::vector<double> upper_left_diag(qubits, 0.); 
  
  //now we Schur decompose the lower right (odd/odd) block
  std::vector<double> lower_right(qubits*qubits);
  for(int i = 0; i < qubits; i++){
    upper_left_diag[i] = A[dense_fortran(2*i+1, 2*i+1, 2*qubits)];
					 
    for(int j = 0; j < qubits; j++){
      //remember we add one 1 to change to fortran indexing
      lower_right[dense_fortran(i+1,j+1, qubits)] = A[dense_fortran(2*i+1+1, 2*j+1+1, 2*qubits)]*upper_left_diag[i];
    }
  }
  

  std::vector<double> workopt(1);
  int sdim = 0;
  std::vector<double> schurvectors(qubits*qubits);
  std::vector<double> eigenvalues_r(qubits);
  std::vector<double> eigenvalues_i(qubits);
  std::vector<int> bwork(qubits);
  int32_t info;

  int MINUS_1 = -1;
  LAPACK_dgees("V", "S", &selectfn, &qubits, &lower_right[0], &qubits, &sdim,
               &eigenvalues_r[0], &eigenvalues_i[0], &schurvectors[0],
               &qubits, &workopt[0], &MINUS_1,  &bwork[0], &info);


  int lwork = (int)workopt[0];
  std::vector<double> work(lwork);
  LAPACK_dgees("V", "S", &selectfn, &qubits, &lower_right[0], &qubits, &sdim,
               &eigenvalues_r[0], &eigenvalues_i[0], &schurvectors[0],
               &qubits, &work[0], &lwork, &bwork[0], &info);
  
  //A = Z*T*(Z**T).
  //where Z is the matrix of schur vectors
  //A is what was in lower_right before dgees, T is what is now in lower_right

  std::fill(Q1.begin(), Q1.end(), 0.);
  std::fill(Q2.begin(), Q2.end(), 0.);

  for(int i = 0; i < qubits; i++){
    for(int j = 0; j < qubits; j++){
      Q1[dense_fortran(2*j+1, 2*i+1, 2*qubits)] = schurvectors[dense_fortran(i+1, j+1, qubits)]*upper_left_diag[i];
      Q1[dense_fortran(2*j+2, 2*i+2, 2*qubits)] = schurvectors[dense_fortran(i+1, j+1, qubits)]*upper_left_diag[i];
      
      Q2[dense_fortran(2*i+1, 2*j+1, 2*qubits)] = schurvectors[dense_fortran(i+1, j+1, qubits)];
      Q2[dense_fortran(2*i+2, 2*j+2, 2*qubits)] = schurvectors[dense_fortran(i+1, j+1, qubits)];
    }
  }

  A = matmul_square_double(matmul_square_double(Q1, A, 2*qubits), Q2, 2*qubits);

  for(auto it = left_ops.rbegin(); it != left_ops.rend(); ++it){
    apply_right(*it, Q1, qubits);
  }

  for(auto it = right_ops.rbegin(); it != right_ops.rend(); ++it){
    apply_left(*it, Q2, qubits);
  }

  return A;
}

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
  double tau = 2;
  LAPACK_dlarf("L",
	       &n, // number of rows
	       &n, //cols
	       &h.w[1],
	       &inc, //incv		
	       &tau, //tau	
	       &m[0], 
	       &n,
	       &work[0]);	
  LAPACK_dlarf("L",
	       &n, // number of rows
	       &n, //cols
	       &h.w[0],
	       &inc, //incv
	       &tau, //tau
	       &m[0], 
	       &n,
	       &work[0]);  
}

void apply_right(SymplecticHouseholder h, std::vector<double> &m, int qubits){
  //a symplectic Householder operation is essentially two normal Householder operations
  std::vector<double> work(2*qubits);
  int n = 2*qubits;
  int inc = 1;
  double tau = 2;
  LAPACK_dlarf("R",
	       &n, // number of rows
	       &n, //cols
	       &h.w[1],
	       &inc, 
	       &tau, 
	       &m[0], 
	       &n,
	       &work[0]);	
  LAPACK_dlarf("R",
	       &n, // number of rows
	       &n, //cols
	       &h.w[0],
	       &inc, 
	       &tau, 
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


/*
std::vector<double> symplectic_givens(int qubits, double theta, int k){
  std::vector<double> givens(4*qubits*qubits);

  std::fill(givens.begin(), givens.end(), 0.);
  for(int i = 0; i < 2*qubits; i++){
    givens[dense_fortran(i+1, i+1, 2*qubits)] = 1;
  }

  givens[dense_fortran(k+1, k+1, 2*qubits)] = cos(theta);
  givens[dense_fortran(qubits+k+1, qubits+k+1, 2*qubits)] = cos(theta);

  givens[dense_fortran(qubits+k+1, k+1, 2*qubits)] =-sin(theta);
  givens[dense_fortran(k+1, qubits+k+1, 2*qubits)] = sin(theta);

  return givens;

}

std::vector<double> symplectic_householder(int qubits, std::vector<double> v){
  std::vector<double> householder(4*qubits*qubits);
  std::fill(householder.begin(), householder.end(), 0.);

  double square_norm = 0;

  for(int i = 0; i < qubits; i++){
    square_norm += v[i]*v[i];
  }

  for(int i = 0; i < qubits; i++){
    householder[dense_fortran(i+1, i+1, 2*qubits)] += 1;
    householder[dense_fortran(i+qubits+1, i+qubits+1, 2*qubits)] += 1;
    for(int j = 0; j < qubits; j++){
      householder[dense_fortran(i+1, j+1, 2*qubits)] -= 2*v[i]*v[j]/square_norm;
      householder[dense_fortran(i+qubits+1, j+qubits+1, 2*qubits)] -= 2*v[i]*v[j]/square_norm;
    }
  }

  return householder;

}

std::vector<double> symplectic_householder_to_zero_below_k(int qubits, std::vector<double> v, int k){
  double norm2 = 0;
  for(int i = 0; i < qubits; i++){
    norm2 += v[i]*v[i];
  }
  v[k] += sqrt(norm2);

  return symplectic_householder(qubits, v);
}



std::vector<double> symplectic_givens_reshuffled(int qubits, double theta, int k){
  std::vector<double> givens(4*qubits*qubits);

  std::fill(givens.begin(), givens.end(), 0.);
  for(int i = 0; i < 2*qubits; i++){
    givens[dense_fortran(i+1, i+1, 2*qubits)] = 1;
  }

  givens[dense_fortran(2*k+1, 2*k+1, 2*qubits)] = cos(theta);
  givens[dense_fortran(2*k+2, 2*k+2, 2*qubits)] = cos(theta);

  givens[dense_fortran(2*k+2, 2*k+1, 2*qubits)] =-sin(theta);
  givens[dense_fortran(2*k+1, 2*k+2, 2*qubits)] = sin(theta);

  return givens;

}

std::vector<double> symplectic_householder_reshuffled(int qubits, std::vector<double> v){
  std::vector<double> householder(4*qubits*qubits);
  std::fill(householder.begin(), householder.end(), 0.);

  double square_norm = 0;

  for(int i = 0; i < qubits; i++){
    square_norm += v[i]*v[i];
  }

  for(int i = 0; i < qubits; i++){
    householder[dense_fortran(i+1, i+1, 2*qubits)] += 1;
    householder[dense_fortran(i+qubits+1, i+qubits+1, 2*qubits)] += 1;
    for(int j = 0; j < qubits; j++){
      householder[dense_fortran(2*i+1, 2*j+1, 2*qubits)] -= 2*v[i]*v[j]/square_norm;
      householder[dense_fortran(2*i+2, 2*j+2, 2*qubits)] -= 2*v[i]*v[j]/square_norm;
    }
  }

  return householder;

}

std::vector<double> symplectic_householder_to_zero_below_k_reshuffled(int qubits, std::vector<double> v, int k){
  double norm2 = 0;
  for(int i = 0; i < qubits; i++){
    norm2 += v[i]*v[i];
  }
  v[k] += sqrt(norm2);
  
  return symplectic_householder_reshuffled(qubits, v);
}


*/

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
    std::fill(v.begin(), v.end(), 0);
    for(int i = k; i < qubits; i++){
      v[i] = A[dense_fortran(2*i+2, 2*k+1, 2*qubits)];
    }
    //A = matmul_square_double(CblasNoTrans,CblasNoTrans, symplectic_householder_to_zero_below_k_reshuffled(qubits, v, k), A, 2*qubits);
    left_ops.push_back(SymplecticHouseholder(qubits, v, k));
    apply_left(left_ops.back(), A, qubits);        

    //eliminate diagonal of lower left block
    double theta = atan2(A[dense_fortran(2*k+2, 2*k+1,2*qubits)], A[dense_fortran(2*k+1, 2*k+1,2*qubits)]);
    left_ops.push_back(SymplecticGivens(cos(theta), sin(theta), k));
    apply_left(left_ops.back(), A, qubits);
    //A = matmul_square_double(CblasNoTrans,CblasNoTrans, symplectic_givens_reshuffled(qubits, theta, k), A, 2*qubits);

    //eliminate below diagonal of upper left block
    std::fill(v.begin(), v.end(), 0);
    for(int i = k; i < qubits; i++){
      v[i] = A[dense_fortran(2*i+1, 2*k+1, 2*qubits)];
    }
    //A = matmul_square_double(CblasNoTrans,CblasNoTrans, symplectic_householder_to_zero_below_k_reshuffled(qubits, v, k), A, 2*qubits);
    left_ops.push_back(SymplecticHouseholder(qubits, v, k));
    apply_left(left_ops.back(), A, qubits);
    //eliminate (from the right) right of diagonal of lower left block
    if(k+1 < qubits - 1){
      std::fill(v.begin(), v.end(), 0);
      for(int i = k+1; i < qubits; i++){
        v[i] = A[dense_fortran(2*k+2, 2*i+1, 2*qubits)];
      }
      //A = matmul_square_double(CblasNoTrans,CblasNoTrans,  A, symplectic_householder_to_zero_below_k_reshuffled(qubits, v, k+1), 2*qubits);
      right_ops.push_back(SymplecticHouseholder(qubits, v, k+1));
      apply_right(right_ops.back(), A, qubits);
    }

    //eliminate the remaining above diagonal of lower left block
    theta = atan2(A[dense_fortran(2*k+2, 2*(k+1)+1,2*qubits)], A[dense_fortran(2*k+2, 2*(k+1)+2,2*qubits)]);
    //A = matmul_square_double(CblasNoTrans,CblasNoTrans, A, symplectic_givens_reshuffled(qubits, theta, k+1), 2*qubits);
    right_ops.push_back(SymplecticGivens(cos(theta), sin(theta), k+1));
    apply_right(right_ops.back(), A, qubits);
    //eliminate (from right) right of diagonal in lower right block
    if(k+1 < qubits - 1){
      std::fill(v.begin(), v.end(), 0);
      for(int i = k+1; i < qubits; i++){
        v[i] = A[dense_fortran(2*k+2, 2*i+1+1, 2*qubits)];
      }

      //A = matmul_square_double(CblasNoTrans,CblasNoTrans,  A, symplectic_householder_to_zero_below_k_reshuffled(qubits, v, k+1), 2*qubits);
      right_ops.push_back(SymplecticHouseholder(qubits, v, k+1));
      apply_right(right_ops.back(), A, qubits);

    }
  }
  //finally delete the lower right element of the lower left block
  double theta = atan2(A[dense_fortran(2*qubits-1+1, 2*qubits - 2 + 1,2*qubits)], A[dense_fortran(2*qubits - 2 + 1, 2*qubits - 2 + 1,2*qubits)]);
  //A = matmul_square_double(CblasNoTrans,CblasNoTrans,  symplectic_givens_reshuffled(qubits, theta, qubits-1), A,  2*qubits);
  left_ops.push_back(SymplecticGivens(cos(theta), sin(theta), qubits - 1));
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

  for(auto it = left_ops.rbegin(); it != left_ops.rend(); ++it){
    apply_right(*it, Q1, qubits);
  }

  for(auto it = right_ops.rbegin(); it != right_ops.rend(); ++it){
    apply_left(*it, Q2, qubits);
  }

  return A;
}

/*
int main(){
  std::cout << std::setw(6) << std::fixed << std::showpos;

  std::vector<double> A(8*8);
  /*
    A[dense_fortran(1,1, 8)] = 0.5213857379750627;
    A[dense_fortran(1,2, 8)] = 0.6038418470063296;
    A[dense_fortran(1,3, 8)] = 0.47094179732225394;
    A[dense_fortran(1,4, 8)] = 0.20324794254467882;
    A[dense_fortran(1,5, 8)] = 0.5287590256200526;
    A[dense_fortran(1,6, 8)] = 0.19103628008078877;
    A[dense_fortran(1,7, 8)] = 0.2815455986418517;
    A[dense_fortran(1,8, 8)] = 0.753681552191594;
    A[dense_fortran(2,1, 8)] = 0.5516717767312141;
    A[dense_fortran(2,2, 8)] = 0.8637220757083885;
    A[dense_fortran(2,3, 8)] = 0.8053722209059218;
    A[dense_fortran(2,4, 8)] = 0.24837266320613882;
    A[dense_fortran(2,5, 8)] = 0.18985741208154028;
    A[dense_fortran(2,6, 8)] = 0.9839955818921721;
    A[dense_fortran(2,7, 8)] = 0.669997165946232;
    A[dense_fortran(2,8, 8)] = 0.2803828299787884;
    A[dense_fortran(3,1, 8)] = 0.20391323427420127;
    A[dense_fortran(3,2, 8)] = 0.6250646854524542;
    A[dense_fortran(3,3, 8)] = 0.6526043158620559;
    A[dense_fortran(3,4, 8)] = 0.8988075287484649;
    A[dense_fortran(3,5, 8)] = 0.974763780707167;
    A[dense_fortran(3,6, 8)] = 0.15393236950220446;
    A[dense_fortran(3,7, 8)] = 0.6990892753701047;
    A[dense_fortran(3,8, 8)] = 0.44724144689102074;
    A[dense_fortran(4,1, 8)] = 0.01751320910460452;
    A[dense_fortran(4,2, 8)] = 0.29102490559414307;
    A[dense_fortran(4,3, 8)] = 0.3812366109308799;
    A[dense_fortran(4,4, 8)] = 0.3210279121387014;
    A[dense_fortran(4,5, 8)] = 0.9425446680115734;
    A[dense_fortran(4,6, 8)] = 0.7026669725351259;
    A[dense_fortran(4,7, 8)] = 0.1364503186469025;
    A[dense_fortran(4,8, 8)] = 0.3432090734659071;
    A[dense_fortran(5,1, 8)] = 0.8119946025956372;
    A[dense_fortran(5,2, 8)] = 0.1484940005253066;
    A[dense_fortran(5,3, 8)] = 0.05932569150816602;
    A[dense_fortran(5,4, 8)] = 0.31441663418966115;
    A[dense_fortran(5,5, 8)] = 0.4201564531486972;
    A[dense_fortran(5,6, 8)] = 0.8080177080661693;
    A[dense_fortran(5,7, 8)] = 0.009507586149987812;
    A[dense_fortran(5,8, 8)] = 0.45408378825522866;
    A[dense_fortran(6,1, 8)] = 0.5586869855327257;
    A[dense_fortran(6,2, 8)] = 0.002888633811329533;
    A[dense_fortran(6,3, 8)] = 0.2977575692859733;
    A[dense_fortran(6,4, 8)] = 0.05379910883697381;
    A[dense_fortran(6,5, 8)] = 0.5676687538421876;
    A[dense_fortran(6,6, 8)] = 0.9405581496364512;
    A[dense_fortran(6,7, 8)] = 0.7242737167653064;
    A[dense_fortran(6,8, 8)] = 0.85637808998257;
    A[dense_fortran(7,1, 8)] = 0.5443156497493541;
    A[dense_fortran(7,2, 8)] = 0.37965068970017857;
    A[dense_fortran(7,3, 8)] = 0.6044247282655253;
    A[dense_fortran(7,4, 8)] = 0.7346039924282802;
    A[dense_fortran(7,5, 8)] = 0.9884081234082996;
    A[dense_fortran(7,6, 8)] = 0.892240908774465;
    A[dense_fortran(7,7, 8)] = 0.5196417809777079;
    A[dense_fortran(7,8, 8)] = 0.20820683854727529;
    A[dense_fortran(8,1, 8)] = 0.2989405121930696;
    A[dense_fortran(8,2, 8)] = 0.8355254093543386;
    A[dense_fortran(8,3, 8)] = 0.18450649293163557;
    A[dense_fortran(8,4, 8)] = 0.18375199251705476;
    A[dense_fortran(8,5, 8)] = 0.23502814077227874;
    A[dense_fortran(8,6, 8)] = 0.6581885823038478;
    A[dense_fortran(8,7, 8)] = 0.5167310218031672;
    A[dense_fortran(8,8, 8)] = 0.8238572345764841;
  

  A[dense_fortran(1,1, 8)] = 0.7389932910989898;
  A[dense_fortran(1,2, 8)] = 0.3302532851899326;
  A[dense_fortran(1,3, 8)] = -0.12909838054541564;
  A[dense_fortran(1,4, 8)] = 0.31592866006122394;
  A[dense_fortran(1,5, 8)] = -0.178810710038615;
  A[dense_fortran(1,6, 8)] = -0.38891564655696226;
  A[dense_fortran(1,7, 8)] = -0.20971284006903906;
  A[dense_fortran(1,8, 8)] = 0.033708274551741024;
  A[dense_fortran(2,1, 8)] = -0.1847935187649797;
  A[dense_fortran(2,2, 8)] = 0.8542327961563694;
  A[dense_fortran(2,3, 8)] = 0.39150725054419583;
  A[dense_fortran(2,4, 8)] = -0.12197486971078803;
  A[dense_fortran(2,5, 8)] = -0.07094194237863863;
  A[dense_fortran(2,6, 8)] = 0.04612306174202503;
  A[dense_fortran(2,7, 8)] = 0.22989220056277238;
  A[dense_fortran(2,8, 8)] = -0.08928252590054543;
  A[dense_fortran(3,1, 8)] = 0.22524765363862742;
  A[dense_fortran(3,2, 8)] = -0.3648557033566148;
  A[dense_fortran(3,3, 8)] = 0.8644275016279775;
  A[dense_fortran(3,4, 8)] = 0.1640758287419152;
  A[dense_fortran(3,5, 8)] = -0.002989330683272739;
  A[dense_fortran(3,6, 8)] = -0.11491868335596124;
  A[dense_fortran(3,7, 8)] = 0.1329364333529444;
  A[dense_fortran(3,8, 8)] = -0.10535982116303115;
  A[dense_fortran(4,1, 8)] = -0.24966601335028812;
  A[dense_fortran(4,2, 8)] = 0.06299488967487758;
  A[dense_fortran(4,3, 8)] = -0.07342662846641931;
  A[dense_fortran(4,4, 8)] = 0.9164784404259237;
  A[dense_fortran(4,5, 8)] = 0.07752207959151908;
  A[dense_fortran(4,6, 8)] = 0.22225865085813187;
  A[dense_fortran(4,7, 8)] = 0.17485883421869947;
  A[dense_fortran(4,8, 8)] = 0.048888971244558856;
  A[dense_fortran(5,1, 8)] = 0.2016069061283215;
  A[dense_fortran(5,2, 8)] = 0.13592427519000058;
  A[dense_fortran(5,3, 8)] = 0.11378753144475455;
  A[dense_fortran(5,4, 8)] = -0.04358079898555901;
  A[dense_fortran(5,5, 8)] = 0.8449726823870746;
  A[dense_fortran(5,6, 8)] = 0.16884662935873485;
  A[dense_fortran(5,7, 8)] = -0.18250914399851226;
  A[dense_fortran(5,8, 8)] = 0.38760128043344344;
  A[dense_fortran(6,1, 8)] = 0.3428921610733557;
  A[dense_fortran(6,2, 8)] = 0.0034390544792561816;
  A[dense_fortran(6,3, 8)] = 0.07223783510378813;
  A[dense_fortran(6,4, 8)] = -0.06248640818877897;
  A[dense_fortran(6,5, 8)] = -0.3784961945784593;
  A[dense_fortran(6,6, 8)] = 0.8214703589747234;
  A[dense_fortran(6,7, 8)] = -0.09076005419945797;
  A[dense_fortran(6,8, 8)] = 0.21674864550255307;
  A[dense_fortran(7,1, 8)] = 0.24522284507520753;
  A[dense_fortran(7,2, 8)] = -0.07495333007426846;
  A[dense_fortran(7,3, 8)] = -0.17801100014845792;
  A[dense_fortran(7,4, 8)] = -0.11002319726153498;
  A[dense_fortran(7,5, 8)] = 0.0196478849910088;
  A[dense_fortran(7,6, 8)] = -0.07937755126351302;
  A[dense_fortran(7,7, 8)] = 0.875443950984513;
  A[dense_fortran(7,8, 8)] = 0.3425869059230192;
  A[dense_fortran(8,1, 8)] = -0.29718012209809475;
  A[dense_fortran(8,2, 8)] = -0.005081321114341234;
  A[dense_fortran(8,3, 8)] = 0.16605867115726053;
  A[dense_fortran(8,4, 8)] = 0.023423543367112745;
  A[dense_fortran(8,5, 8)] = -0.3152001256178392;
  A[dense_fortran(8,6, 8)] = -0.272742627414048;
  A[dense_fortran(8,7, 8)] = -0.216733262829974;
  A[dense_fortran(8,8, 8)] = 0.8141382212744724;

  print_fortran(reshuffled(A,4),8);

  std::vector<double> Q1(8*8);
  std::vector<double> Q2(8*8);

  

  std::cout << std::endl;
  std::cout << std::setw(6) << std::fixed << std::showpos;
  std::vector<double> B = symplectic_orthogonal_factorize(4, A, Q1, Q2);
  print_fortran(reshuffled(A,4), 8);
  print_fortran(reshuffled(matmul_square_double(matmul_square_double(Q1, A, 8), Q2, 8), 4), 8);

  
  //print_fortran(A, 2*4);

  //std::cout << std::endl;

  //print_fortran(reshuffled(B, 4), 2*4);

  return 0;
}
*/

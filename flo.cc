#include "ff.h"
#include <iomanip>


using namespace std::complex_literals;

//for some reason this isn't already a thing??
std::complex<double> operator * (const int & a, const std::complex<double> & b){
  return std::complex<double>(b.real() * a, b.imag() * a);
}

std::complex<double> operator * (const std::complex<double> & b, const int & a){
  return std::complex<double>(b.real() * a, b.imag() * a);
}

std::vector<double> reorder_vec(std::vector<int> x){
  //we compute a permutation matrix P so that all the non-zero entries of Px are at the start
  int one_count = 0;
  std::vector<std::pair<int,int> > perm;

  for(int i = 0; i < x.size(); i++){
    if(x[i] != 0){
      //permute this element to the start
      perm.push_back(std::pair(i, one_count));
      one_count += 1;
    }
  }
  std::vector<double> P(x.size()*x.size());

  std::fill(P.begin(), P.end(), 0.0);
  for(int i = 0; i < x.size();i++){
    P[dense_fortran(i+1,i+1,x.size())] = 1;
  }
  for(const std::pair<int,int> &pair: perm){
    for(int i = 0; i < x.size();i++){
      std::swap(P[dense_fortran(pair.first+1,i+1,x.size())],
                P[dense_fortran(pair.second+1,i+1,x.size())]);
    }
  }

  return P;
}



std::complex<double> cb_inner_prod(int qubits, std::vector<int> x, std::vector<double> R, std::complex<double> phase, std::vector<double> l){
  std::complex<double> prefactor = phase*(1-2*(qubits % 2));
  for(const double& li : l){
    prefactor *= (cos(li/2.)*cos(li/2.));
  }
  std::vector<double> perm = reorder_vec(x);

  int x_vec_weight = 0;
  for(const int& i: x){
    if(i != 0){
      x_vec_weight += 1;
    }
  }
  //std::cout << "perm = " << std::endl;
  //print_fortran(perm, 2*qubits);

  std::vector<double> x_vec_matrix = std::vector<double>(2*qubits*2*qubits);
  std::fill(x_vec_matrix.begin(), x_vec_matrix.end(),0);
  for(int i = 0; i < x_vec_weight-1; i++){
    x_vec_matrix[dense_fortran(i+1, i+1, 2*qubits)] = (1 - 2*((x_vec_weight+1) % 2));
  }
  if(x_vec_weight > 0){
    x_vec_matrix[dense_fortran(x_vec_weight, x_vec_weight, 2*qubits)] = 1 - 2*((x_vec_weight+1) % 2);
  }
  for(int i = x_vec_weight; i <2*qubits ; i++){
    x_vec_matrix[dense_fortran(i+1, i+1, 2*qubits)] = 1 - 2*(x_vec_weight % 2);
  }

  //this version is real
  //the correct version should be i times this
  std::vector<double> realM(2*qubits*2*qubits);
  std::fill(realM.begin(), realM.end(),0);
  for(int i = 0; i < qubits; i++){
    realM[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = -1;
    realM[dense_fortran(2*i+2, 2*i+1, 2*qubits)] =  1;
  }
  //std::cout << "realM1 = " << std::endl;
  //print_fortran(realM,2*qubits);
  //replace realM with perm @ realM @ perm.T
  matrix_conjugate_inplace_double(realM,perm,2*qubits);
  //std::cout << "realMFinal = " << std::endl;
  //print_fortran(realM,2*qubits);

  std::vector<double> T(2*qubits*2*qubits);
  std::fill(T.begin(), T.end(), 0);
  for(int i = 0; i < l.size(); i++){
    double val = tan(l[i]/2.);
    T[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = val;
    T[dense_fortran(4*i+3, 4*i+1, 2*qubits)] = -val;
    T[dense_fortran(4*i+2, 4*i+4, 2*qubits)] = -val;
    T[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = val;
  }
  //std::cout << "T = " << std::endl;
  //print_fortran(T, 2*qubits);
  std::vector<double> scratch(2*qubits*2*qubits);
  std::fill(scratch.begin(), scratch.end(), 0);
  matmul_square_double(x_vec_matrix, perm, scratch, 2*qubits);
  matmul_square_double(CblasNoTrans, CblasTrans, scratch, R, perm,2*qubits);

  //std::cout << "big conjugating matrix" << std::endl;
  //print_fortran(perm, 2*qubits);

  matrix_conjugate_inplace_double(T, perm, 2*qubits);
  //std::cout << "T'" << std::endl;
  //print_fortran(T, 2*qubits);


  int w = x_vec_weight; //just a shorthand
  int G_dim = 4*qubits+w;
  std::vector<std::complex<double> > G(G_dim*G_dim);
  std::fill(G.begin(), G.end(), 0);
  /*
    std::cout << "w = " << w << " 2*qubits = " << 2*qubits << std::endl;
    realM[dense_fortran(1,1,2*qubits)] = 1;
    realM[dense_fortran(1,2,2*qubits)] = 1;
    realM[dense_fortran(2,1,2*qubits)] = 1;
    realM[dense_fortran(2,2,2*qubits)] = 1;

    realM[dense_fortran(1,3,2*qubits)] = 2;
    realM[dense_fortran(1,4,2*qubits)] = 2;
    realM[dense_fortran(2,3,2*qubits)] = 2;
    realM[dense_fortran(2,4,2*qubits)] = 2;

    realM[dense_fortran(3,1,2*qubits)] = 3;
    realM[dense_fortran(3,2,2*qubits)] = 3;
    realM[dense_fortran(4,1,2*qubits)] = 3;
    realM[dense_fortran(4,2,2*qubits)] = 3;

    realM[dense_fortran(3,3,2*qubits)] = 4;
    realM[dense_fortran(3,4,2*qubits)] = 4;
    realM[dense_fortran(4,3,2*qubits)] = 4;
    realM[dense_fortran(4,4,2*qubits)] = 4;
    print_fortran(realM, 2*qubits);
  */
  matrix_add_block(G, realM, 1.i,
                   G_dim, 2*qubits,
                   w, w,
                   w, w, //M22
                   2*qubits-w, 2*qubits-w); //M11 has width w so M22 has width the rest
  matrix_add_block(G, realM, 1.i,
                   G_dim, 2*qubits,
                   2*qubits, w,
                   0, w, //M12
                   2*qubits-w, w);
  matrix_add_block(G, realM, 1.i,
                   G_dim, 2*qubits,
                   w,2*qubits,
                   w, 0, //M21
                   w, 2*qubits-w);
  matrix_add_block(G, realM, 1.i,
                   G_dim, 2*qubits,
                   2*qubits, 2*qubits, // stick M11 at 2*qubits
                   0, 0, //M11
                   w, w); //M11 has width w
  //print_fortran(G, G_dim);
  for(int i = 0; i < 2*qubits; i++){
    G[dense_fortran(i+1, 2*qubits+w+i+1, G_dim)]  += 1;
    G[dense_fortran(2*qubits+w+i+1, i+1, G_dim)]  += -1;
  }
  for(int i = 0; i < w; i++){
    G[dense_fortran(i+1, 2*qubits+i+1, G_dim)]  += -1;
    G[dense_fortran(2*qubits+i+1, i+1, G_dim)]  += 1;
  }

  matrix_add_block(G, T, 1,
                   G_dim, 2*qubits,
                   2*qubits+w, 2*qubits+w, // stick T at 2*qubits+x
                   0, 0, //want the whole of T
                   2*qubits, 2*qubits);

  std::complex<double> pfaffian=0;
  int info;
  //print_fortran(G, G_dim);
  //return 0;
  /* Compute the pfaffian using the lower triangle and the Parlett-Reid
     algorithm */
  //print_fortran(G, G_dim);
  info = skpfa(G_dim, &G[0], &pfaffian, "L", "P");

  //std::cout << "prefactor = " << prefactor << std::endl;
  return prefactor*pfaffian;
}

DecomposedPassive decompose_passive_flo_unitary(std::vector<double> R,int qubits, std::complex<double> phase){
  //split alpha into x\otimes I + y\otimes i\sigma_y
  //then create the complex matrix x +iy
  //we use lapack so we need some different types
  using cdouble = std::complex<double>; //__complex__ double;
  const int32_t n = qubits;
  std::vector<cdouble> U(n*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      U[dense_fortran(i+1,j+1,n)] = R[dense_fortran(2*i+1,2*j+1, 2*n)] + (R[dense_fortran(2*i+1,2*j+2, 2*n)]*(1.i));
    }
  }
  std::vector<cdouble> eigenvectors(n*n);
  std::vector<cdouble>  eigenvalues(n);

  int32_t m;
  int32_t isuppz;

  /*
    ZGEES( JOBVS, SORT, SELECT, N, A, LDA, SDIM, W, VS, LDVS, WORK,
    LWORK, RWORK, BWORK, INFO )

    CHARACTER       JOBVS, SORT

    INTEGER         INFO, LDA, LDVS, LWORK, N, SDIM

    LOGICAL         BWORK( * )

    DOUBLE          PRECISION RWORK( * )

    COMPLEX*16    A( LDA, * ), VS( LDVS, * ),   W( * ), WORK( * )

    LOGICAL         SELECT

    EXTERNAL        SELECT
  */
  std::vector<double> rwork(n);
  int32_t info;
  int32_t MINUS_1 = -1;
  int sdim = 0;

  std::vector<cdouble> workopt(1);
  LAPACK_zgees("V", "N", NULL, &n, &U[0], &n, &sdim,
               &eigenvalues[0], &eigenvectors[0],
               &n, &workopt[0], &MINUS_1, &rwork[0], NULL, &info);


  int lwork = (int)workopt[0].real();
  std::vector<cdouble> work(lwork);
  LAPACK_zgees("V", "N", NULL, &n, &U[0], &n, &sdim,
               &eigenvalues[0], &eigenvectors[0],
               &n, &work[0], &lwork, &rwork[0], NULL, &info);

  //std::vector<double> newR(2*n*2*n);
  std::vector<double> V(2*n*2*n);
  /*
  for(int i = 0; i < n; i++){
    newR[dense_fortran(2*i+1, 2*i+1, 2*n)] = creal(eigenvalues[i]);
    newR[dense_fortran(2*i+2, 2*i+2, 2*n)] = creal(eigenvalues[i]);

    newR[dense_fortran(2*i+1, 2*i+2, 2*n)] = cimag(eigenvalues[i]);
    newR[dense_fortran(2*i+2, 2*i+1, 2*n)] =-cimag(eigenvalues[i]);
    }*/
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      /*
      V[dense_fortran(2*j+1, 2*i+1, 2*n)] = creal(eigenvectors[dense_fortran(i+1, j+1, n)]);
      V[dense_fortran(2*j+2, 2*i+2, 2*n)] = creal(eigenvectors[dense_fortran(i+1, j+1, n)]);
      V[dense_fortran(2*j+1, 2*i+2, 2*n)] =-cimag(eigenvectors[dense_fortran(i+1, j+1, n)]);
      V[dense_fortran(2*j+2, 2*i+1, 2*n)] =cimag(eigenvectors[dense_fortran(i+1, j+1, n)]);
      */

      V[dense_fortran(2*j+1, 2*i+1, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].real();
      V[dense_fortran(2*j+2, 2*i+2, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].real();
      V[dense_fortran(2*j+1, 2*i+2, 2*n)] =-eigenvectors[dense_fortran(i+1, j+1, n)].imag();
      V[dense_fortran(2*j+2, 2*i+1, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].imag();
    }
  }
  /*
    matrix_conjugate_inplace_double(R, V, 2*n);
    for(int i = 0; i < 2*n; i++){
    for(int j = 0; j < 2*n; j++){
    if(abs(R[dense_fortran(i+1,j+1,2*n)]) < 1e-13){
    R[dense_fortran(i+1,j+1,2*n)] = 0;
    }
    }
    }
    print_fortran(R, 2*n);
    std::cout << std::endl;
  */
  //print_fortran(newR, 2*n);
  //we now compute the log of newR
  //which is block diagonal with 2x2 orthogonal blocks
  //so this is pretty easy
  std::vector<double> lambdas(n);
  double sum = 0;
  for(int i = 0; i < n; i++){
    lambdas[i] = std::atan2((eigenvalues[i].imag()), (eigenvalues[i].real()));
    sum += lambdas[i]/2.;
  }
  std::complex<double> new_phase = std::exp(sum*1.i);
  //std::cout << phase << ", " << new_phase << ", " << phase/new_phase << std::endl;

  //we might be wrong by a factor of minus 1 in new phase
  //if we are we have to add 2pi to lambda_0

  if(abs(phase - new_phase) > 1e-10){
    //std::cout << "phases " << phase << " and " << new_phase << "don't match!"<<std::endl;
    lambdas[0] += 2*M_PI;
    new_phase = -new_phase;
  }
  DecomposedPassive p;
  p.phase = new_phase;
  p.l = lambdas;
  p.R = V;


  return p;
}
/*
  std::complex<double> inner_prod(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2)
  {

  //first compute K_1^dagger K_2
  std::vector<double> K(2*qubits*2*qubits);
  matmul_square_double(CblasTrans, CblasNoTrans, K1, K2, K, 2*qubits);
  //std::cout << "K = "<<std::endl;
  //print_fortran(K, 2*qubits);
  DecomposedPassive p = decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  //std::cout << "expected phase "<<std::conj(phase1)*phase2<< ", found "<< p.phase << std::endl;

  std::complex<double> prefactor = 1;

  std::vector<double> T(2*qubits*2*qubits);
  std::fill(T.begin(), T.end(), 0);
  for(int i = 0; i < qubits; i++){
  prefactor *= (std::cos(p.l[i]/2)*(-1.i));
  T[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = std::tan(p.l[i]/2);
  T[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -std::tan(p.l[i]/2);
  }
  matrix_conjugate_inplace_double(T, p.R, 2*qubits, CblasTrans);
  //std::cout << "T:" << std::endl;
  //print_fortran(T, 2*qubits);

  //this K now gets conjugated by A1
  //K <- A1^dagger K A^1
  //we have to work out the special orthogonal matrix
  //that implements
  //A1^\dagger = exp(-1/2 sum_j A1[j]*(c[4j]@c[4j+2] - c[4j+1]@c[4j+3]))

  std::vector<double> A1R(2*qubits*2*qubits);
  std::fill(A1R.begin(), A1R.end(), 0);
  for(int i = 0; i < 2*qubits; i++){
  A1R[dense_fortran(i+1,i+1, 2*qubits)] = 1;
  }
  for(int i = 0; i < qubits/2; i++){
  A1R[dense_fortran(4*i+1,4*i+1, 2*qubits)] = std::cos(-A1[i]);
  A1R[dense_fortran(4*i+2,4*i+2, 2*qubits)] = std::cos(-A1[i]);
  A1R[dense_fortran(4*i+3,4*i+3, 2*qubits)] = std::cos(-A1[i]);
  A1R[dense_fortran(4*i+4,4*i+4, 2*qubits)] = std::cos(-A1[i]);

  A1R[dense_fortran(4*i+1,4*i+2+1, 2*qubits)] = std::sin(-A1[i]);
  A1R[dense_fortran(4*i+1+1,4*i+3+1, 2*qubits)] = std::sin(A1[i]);
  A1R[dense_fortran(4*i+3+1,4*i+1+1, 2*qubits)] = std::sin(-A1[i]);
  A1R[dense_fortran(4*i+2+1,4*i+1, 2*qubits)] = std::sin(A1[i]);
  }
  //std::cout << "A1R:" << std::endl;
  //print_fortran(A1R, 2*qubits);
  matrix_conjugate_inplace_double(T, A1R, 2*qubits, CblasNoTrans);
  std::vector<double> T2(2*qubits*2*qubits);
  std::fill(T2.begin(), T2.end(), 0);
  for(int i = 0; i < qubits/2; i++){
  double angle = (-A1[i] + A2[i])/2;
  prefactor *= (std::cos(angle)*std::cos(angle));
  double val = std::tan(angle);
  T2[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = val;
  T2[dense_fortran(4*i+3, 4*i+1, 2*qubits)] = -val;
  T2[dense_fortran(4*i+2, 4*i+4, 2*qubits)] = -val;
  T2[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = val;
  }

  int G_dim = 6*qubits;
  std::vector<std::complex<double> > G(G_dim*G_dim);
  std::fill(G.begin(), G.end(),0);

  for(int i = 0; i < qubits; i++){
  G[dense_fortran(2*i+1, 2*i+2, G_dim)] = -1.j;
  G[dense_fortran(2*i+2, 2*i+1, G_dim)] = 1.j;
  }

  matrix_add_block(G, T, 1,
  G_dim, 2*qubits,
  2*qubits,2*qubits,
  0,0,
  2*qubits,2*qubits);
  matrix_add_block(G, T2, 1,
  G_dim, 2*qubits,
  4*qubits,4*qubits,
  0,0,
  2*qubits,2*qubits);

  //add identity blocks
  for(int i = 0; i < 2*qubits; i++){
  G[dense_fortran(i+1, 2*qubits+i+1, G_dim)] = 1;
  G[dense_fortran(2*qubits+i+1, i+1, G_dim)] = -1;

  G[dense_fortran(i+1, 4*qubits+i+1, G_dim)] = -1;
  G[dense_fortran(4*qubits+i+1, i+1, G_dim)] = 1;

  G[dense_fortran(2*qubits+i+1, 4*qubits+i+1, G_dim)] = 1;
  G[dense_fortran(4*qubits+i+1, 2*qubits+i+1, G_dim)] = -1;
  }
  //std::cout << "G = " << std::endl;
  //print_fortran(G, G_dim);
  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(G_dim, &G[0], &pfaffian, "U", "P");
  //std::cout << "info: " <<info <<std::endl;
  std::cout << "prefactor: "<< prefactor <<std::endl;
  return prefactor*pfaffian;
  }
*/
/*

  std::complex<double> inner_prod_fast(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2)
  {
  //first compute K_1^dagger K_2
  std::vector<double> K(2*qubits*2*qubits);
  matmul_square_double(CblasTrans, CblasNoTrans, K1, K2, K, 2*qubits);
  //std::cout << "K = "<<std::endl;
  //print_fortran(K, 2*qubits);
  DecomposedPassive p = decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  //std::cout << "expected phase "<<std::conj(phase1)*phase2<< ", found "<< p.phase << std::endl;

  std::complex<double> prefactor = 1;

  std::vector<double> T(2*qubits*2*qubits);
  std::fill(T.begin(), T.end(), 0);
  for(int i = 0; i < qubits; i++){
  prefactor *= (std::cos(p.l[i]/2)*(-1.i));
  T[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = std::tan(p.l[i]/2);
  T[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -std::tan(p.l[i]/2);
  }
  matrix_conjugate_inplace_double(T, p.R, 2*qubits, CblasTrans);
  //std::cout << "T:" << std::endl;
  //print_fortran(T, 2*qubits);

  //this K now gets conjugated by A1
  //K <- A1^dagger K A^1
  //we have to work out the special orthogonal matrix
  //that implements
  //A1^\dagger = exp(-1/2 sum_j A1[j]*(c[4j]@c[4j+2] - c[4j+1]@c[4j+3]))

  std::vector<double> A1R(2*qubits*2*qubits);
  std::fill(A1R.begin(), A1R.end(), 0);
  for(int i = 0; i < 2*qubits; i++){
  A1R[dense_fortran(i+1,i+1, 2*qubits)] = 1;
  }
  for(int i = 0; i < qubits/2; i++){
  A1R[dense_fortran(4*i+1,4*i+1, 2*qubits)] = std::cos(-A1[i]);
  A1R[dense_fortran(4*i+2,4*i+2, 2*qubits)] = std::cos(-A1[i]);
  A1R[dense_fortran(4*i+3,4*i+3, 2*qubits)] = std::cos(-A1[i]);
  A1R[dense_fortran(4*i+4,4*i+4, 2*qubits)] = std::cos(-A1[i]);

  A1R[dense_fortran(4*i+1,4*i+2+1, 2*qubits)] = std::sin(-A1[i]);
  A1R[dense_fortran(4*i+1+1,4*i+3+1, 2*qubits)] = std::sin(A1[i]);
  A1R[dense_fortran(4*i+3+1,4*i+1+1, 2*qubits)] = std::sin(-A1[i]);
  A1R[dense_fortran(4*i+2+1,4*i+1, 2*qubits)] = std::sin(A1[i]);
  }
  //std::cout << "A1R:" << std::endl;
  //print_fortran(A1R, 2*qubits);
  matrix_conjugate_inplace_double(T, A1R, 2*qubits, CblasNoTrans);
  std::vector<std::complex<double> > Tcomplex(2*qubits*2*qubits);
  for(int i = 0; i < 2*qubits; i++){
  for(int j = 0; j < 2*qubits; j++){
  Tcomplex[dense_fortran(i+1,j+1,2*qubits)] = T[dense_fortran(i+1,j+1,2*qubits)];
  }
  }

  std::complex<double> pfaff_prefactor = 1.;
  for(int i = 0; i < qubits/2; i++){
  double angle = (-A1[i] + A2[i])/2;
  //std::cout << "angle["<<i<<"] = " << angle;
  while(angle > M_PI/4.){
  angle -= M_PI/2.;
  prefactor = -prefactor;
  }
  while(angle < -M_PI/4.){
  angle += M_PI/2.;
  prefactor = -prefactor;
  }
  //std::cout << ". Now = " << angle << std::endl;
  prefactor *= (std::cos(angle)*std::cos(angle));
  double t = std::tan(angle);
  //std::cout << "angle = " << angle << ", t = " << t << std::endl;

  pfaff_prefactor *= (1-t*t);

  Tcomplex[dense_fortran(4*i+1,4*i+2,2*qubits)] += -1.i;
  Tcomplex[dense_fortran(4*i+2,4*i+1,2*qubits)] +=  1.i;
  Tcomplex[dense_fortran(4*i+3,4*i+4,2*qubits)] += -1.i;
  Tcomplex[dense_fortran(4*i+4,4*i+3,2*qubits)] +=  1.i;

  std::complex<double> val;
  val = std::cosh(std::atanh(t))*std::sinh(std::atanh(t))*2;

  Tcomplex[dense_fortran(4*i+1,4*i+3,2*qubits)] += val;
  Tcomplex[dense_fortran(4*i+1,4*i+4,2*qubits)] += -val*1.i;
  Tcomplex[dense_fortran(4*i+2,4*i+3,2*qubits)] += -val*1.i;
  Tcomplex[dense_fortran(4*i+2,4*i+4,2*qubits)] += -val;

  Tcomplex[dense_fortran(4*i+3,4*i+1,2*qubits)] += -val;
  Tcomplex[dense_fortran(4*i+3,4*i+2,2*qubits)] += val*1.i;
  Tcomplex[dense_fortran(4*i+4,4*i+1,2*qubits)] += val*1.i;
  Tcomplex[dense_fortran(4*i+4,4*i+2,2*qubits)] += val;
  }
  //print_fortran(Tcomplex, 2*qubits);




  pfaff_prefactor *= std::pow(-1, qubits);
  //std::cout << "G = " << std::endl;
  //print_fortran(G, G_dim);
  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(2*qubits, &Tcomplex[0], &pfaffian, "U", "P");
  //std::cout << "info: " <<info <<std::endl;
  //std::cout << "prefactor: " << prefactor << ", pfaff_prefactor:" << pfaff_prefactor << std::endl;
  //std::cout << "pfafian: " <<pfaffian <<std::endl;
  return pfaff_prefactor * prefactor*pfaffian;
  }
*/

/*
std::complex<double> inner_prod(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2)
{
  //first compute K_1^dagger K_2
  std::vector<double> K(2*qubits*2*qubits);
  matmul_square_double(CblasTrans, CblasNoTrans, K1, K2, K, 2*qubits);
  DecomposedPassive p = decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  //std::cout << "expected phase "<<std::conj(phase1)*phase2<< ", found "<< p.phase << std::endl;

  std::complex<double> prefactor = 1;
    
  
  std::vector<double> T(2*qubits*2*qubits);
  std::fill(T.begin(), T.end(), 0);
  for(int i = 0; i < qubits; i++){
    prefactor *= (std::cos(p.l[i]/2)*(-1.i));
    T[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = std::tan(p.l[i]/2);
    T[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -std::tan(p.l[i]/2);
  }
  matrix_conjugate_inplace_double(T, p.R, 2*qubits, CblasTrans);

  //this K now gets conjugated by A1
  //K <- A1^dagger K A^1
  //we have to work out the special orthogonal matrix
  //that implements
  //A1^\dagger = exp(-1/2 sum_j A1[j]*(c[4j]@c[4j+2] - c[4j+1]@c[4j+3]))

  std::vector<double> A1R(2*qubits*2*qubits);
  std::fill(A1R.begin(), A1R.end(), 0);
  for(int i = 0; i < 2*qubits; i++){
    A1R[dense_fortran(i+1,i+1, 2*qubits)] = 1;
  }
  for(int i = 0; i < qubits/2; i++){
    A1R[dense_fortran(4*i+1,4*i+1, 2*qubits)] = std::cos(-A1[i]);
    A1R[dense_fortran(4*i+2,4*i+2, 2*qubits)] = std::cos(-A1[i]);
    A1R[dense_fortran(4*i+3,4*i+3, 2*qubits)] = std::cos(-A1[i]);
    A1R[dense_fortran(4*i+4,4*i+4, 2*qubits)] = std::cos(-A1[i]);

    A1R[dense_fortran(4*i+1,4*i+2+1, 2*qubits)] = std::sin(-A1[i]);
    A1R[dense_fortran(4*i+1+1,4*i+3+1, 2*qubits)] = std::sin(A1[i]);
    A1R[dense_fortran(4*i+3+1,4*i+1+1, 2*qubits)] = std::sin(-A1[i]);
    A1R[dense_fortran(4*i+2+1,4*i+1, 2*qubits)] = std::sin(A1[i]);
  }

  matrix_conjugate_inplace_double(T, A1R, 2*qubits, CblasNoTrans);
  std::vector<std::complex<double> > Tcomplex(2*qubits*2*qubits);
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      Tcomplex[dense_fortran(i+1,j+1,2*qubits)] = T[dense_fortran(i+1,j+1,2*qubits)];
    }
  }
  //add the -iM
  for(int i = 0; i < qubits/2; i++){
    Tcomplex[dense_fortran(4*i+1,4*i+2,2*qubits)] += -1.i;
    Tcomplex[dense_fortran(4*i+2,4*i+1,2*qubits)] +=  1.i;
    Tcomplex[dense_fortran(4*i+3,4*i+4,2*qubits)] += -1.i;
    Tcomplex[dense_fortran(4*i+4,4*i+3,2*qubits)] +=  1.i;
  }

  //add i times the (4+1)th row/col to the (4+2)th row/col
  //this simplifies the extra term \propto iY\otimes (Z-iX) term arising from the Aitken block diagonalization
  //since Z-iX is equal to
  // 1  -i
  // -i -1
  // i times the first row/col ends up deleting the second row/col
  for(int i = 0; i < qubits/2; i++){
    for(int j = 0; j < 2*qubits; j++){
      Tcomplex[dense_fortran(4*i+2,j+1,2*qubits)] += (1.i)*Tcomplex[dense_fortran(4*i+1,j+1,2*qubits)];
      Tcomplex[dense_fortran(j+1,4*i+2,2*qubits)] += (1.i)*Tcomplex[dense_fortran(j+1,4*i+1,2*qubits)];
    }
  }

  for(int i = 0; i < qubits/2; i++){
    double angle = (-A1[i] + A2[i])/2;

    while(angle > M_PI/4.){
      angle -= M_PI/2.;
      prefactor = -prefactor;
    }
    while(angle < -M_PI/4.){
      angle += M_PI/2.;
      prefactor = -prefactor;
    }

    prefactor *= (std::cos(angle)*std::cos(angle));
    double t = std::tan(angle);

    //multiply the (4i+1)th row/col by (1-t*t)
    for(int j = 0; j < 2*qubits; j++){
      Tcomplex[dense_fortran(4*i+1,j+1,2*qubits)] *= (1-t*t);
      Tcomplex[dense_fortran(j+1,4*i+1,2*qubits)] *= (1-t*t);
    }

    std::complex<double> val = 2*t;

    Tcomplex[dense_fortran(4*i+1,4*i+3,2*qubits)] += val;
    Tcomplex[dense_fortran(4*i+1,4*i+4,2*qubits)] += -val*1.i;

    Tcomplex[dense_fortran(4*i+3,4*i+1,2*qubits)] += -val;
    Tcomplex[dense_fortran(4*i+4,4*i+1,2*qubits)] += val*1.i;
  }
  prefactor *= std::pow(-1, qubits);

  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(2*qubits, &Tcomplex[0], &pfaffian, "U", "P");

  return prefactor*pfaffian;
}
*/
/*
std::complex<double> inner_prod_smooth(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2)
{
  //first compute K_1^dagger K_2
  std::vector<double> K(2*qubits*2*qubits);
  matmul_square_double(CblasTrans, CblasNoTrans, K1, K2, K, 2*qubits);
  DecomposedPassive p = decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  if(std::abs(p.phase - std::conj(phase1)*phase2) > 1e-10){
    std::cout << "expected phase "<<std::conj(phase1)*phase2<< ", found "<< p.phase << std::endl;  
  }
    
  //we have to work out the special orthogonal matrix
  //that implements
  //A1^\dagger = exp(-1/2 sum_j A1[j]*(c[4j]@c[4j+2] - c[4j+1]@c[4j+3]))

  std::vector<double> A1RT(2*qubits*2*qubits);
  std::fill(A1RT.begin(), A1RT.end(), 0);
  for(int i = 0; i < 2*qubits; i++){
    A1RT[dense_fortran(i+1,i+1, 2*qubits)] = 1;
  }
  for(int i = 0; i < qubits/2; i++){
    A1RT[dense_fortran(4*i+1,4*i+1, 2*qubits)] = std::cos(-A1[i]);
    A1RT[dense_fortran(4*i+2,4*i+2, 2*qubits)] = std::cos(-A1[i]);
    A1RT[dense_fortran(4*i+3,4*i+3, 2*qubits)] = std::cos(-A1[i]);
    A1RT[dense_fortran(4*i+4,4*i+4, 2*qubits)] = std::cos(-A1[i]);

    A1RT[dense_fortran(4*i+1,4*i+2+1, 2*qubits)] = -std::sin(-A1[i]);
    A1RT[dense_fortran(4*i+1+1,4*i+3+1, 2*qubits)] = -std::sin(A1[i]);
    A1RT[dense_fortran(4*i+3+1,4*i+1+1, 2*qubits)] = -std::sin(-A1[i]);
    A1RT[dense_fortran(4*i+2+1,4*i+1, 2*qubits)] = -std::sin(A1[i]);
  }

  std::vector<double> KT_A1RT = matmul_square_double(CblasNoTrans, CblasNoTrans, p.R, A1RT, 2*qubits);

  //left multiply this by C1
  //multiplies alternate rows by cos(theta_j)
  for(int i = 0; i < qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      KT_A1RT[dense_fortran(2*i+1, j+1, 2*qubits)] *= std::cos(p.l[i]/2.);
    }
  }

  
  std::vector<double> M(2*qubits*2*qubits);
  std::fill(M.begin(),M.end(),0);
  for(int i = 0; i < qubits; i++){
    M[dense_fortran(2*i+1, 2*i+2,2*qubits)] = 1;
    M[dense_fortran(2*i+2, 2*i+1,2*qubits)] = -1;
  }
  matrix_conjugate_inplace_double(M, KT_A1RT,2*qubits);

  std::vector<std::complex<double> > G(4*qubits*4*qubits);
  std::fill(G.begin(), G.end(), 0);
  int G_dim = 4*qubits;
  //first fill the top right corner of G
  for(int i = 0; i < qubits; i++){
    G[dense_fortran(2*i+1, 2*i+2, G_dim)] =std::sin(p.l[i]/2.);
    G[dense_fortran(2*i+2, 2*i+1, G_dim)] =-std::sin(p.l[i]/2.);
  }
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      G[dense_fortran(i+1, j+1, G_dim)] += (-1.i)*M[dense_fortran(i+1,j+1,2*qubits)];
    }
  }

  //bottom right of G
  for(int i = 0; i < qubits/2;i++){
    double angle = (-A1[i] + A2[i])/2;
    G[dense_fortran(2*qubits+4*i+1, 2*qubits+4*i+2, G_dim)] = (-1.i)*std::cos(angle);
    G[dense_fortran(2*qubits+4*i+1, 2*qubits+4*i+3, G_dim)] = std::sin(angle);
    
    G[dense_fortran(2*qubits+4*i+2, 2*qubits+4*i+1, G_dim)] = (1.i)*std::cos(angle);
    G[dense_fortran(2*qubits+4*i+2, 2*qubits+4*i+4, G_dim)] = -std::sin(angle);
    
    G[dense_fortran(2*qubits+4*i+3, 2*qubits+4*i+1, G_dim)] = -std::sin(angle);
    G[dense_fortran(2*qubits+4*i+3, 2*qubits+4*i+4, G_dim)] = (-1.i)*std::cos(angle);

    G[dense_fortran(2*qubits+4*i+4, 2*qubits+4*i+2, G_dim)] = std::sin(angle);
    G[dense_fortran(2*qubits+4*i+4, 2*qubits+4*i+3, G_dim)] = (1.i)*std::cos(angle);
  }
  //top right
  for(int i = 0; i < 2*qubits; i++){
    
    for(int j = 0; j < 2*qubits; j++){
      int sign = 1-2*(j % 2); 
      G[dense_fortran(i+1, 2*qubits+j+1, G_dim)] += KT_A1RT[dense_fortran(i+1,j+1,2*qubits)];
      G[dense_fortran(i+1, 2*qubits+j+1, G_dim)] += (-1.i)*sign*KT_A1RT[dense_fortran(i+1,j+1+sign,2*qubits)];
    }
  }

  //right multiply top right by C2
  for(int i = 0; i < qubits/2; i++){
    double angle = (-A1[i] + A2[i])/2;
    for(int j = 0; j < 2*qubits; j++){      
      G[dense_fortran(j+1, 2*qubits+4*i+1, G_dim)] *= std::cos(angle);
      G[dense_fortran(j+1, 2*qubits+4*i+4, G_dim)] *= std::cos(angle);
    }
  }
  //bottom left
  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(G_dim, &G[0], &pfaffian, "U", "P");

  return std::pow(-1.i, qubits)*pfaffian*(1-2*((qubits/2)%2));
}
*/


/*
std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, std::vector<int> y, DecomposedPassive &p, std::vector<double> T, std::vector<double> A){

  std::complex<double> prefactor = 1;

  for(int i = 0; i < qubits; i++){
    prefactor *= (std::cos(p.l[i]/2)*(-1.i));
  }

  //this K now gets conjugated by A1
  //K <- A1^dagger K A^1
  //we have to work out the special orthogonal matrix
  //that implements
  //A1^\dagger = exp(-1/2 sum_j A1[j]*(c[4j]@c[4j+2] - c[4j+1]@c[4j+3]))

  std::vector<double> A1R(2*qubits*2*qubits);
  std::fill(A1R.begin(), A1R.end(), 0);
  for(int i = 0; i < 2*qubits; i++){
    A1R[dense_fortran(i+1,i+1, 2*qubits)] = 1;
  }
  for(int i = 0; i < qubits/2; i++){
    if(y[i] == 1){
      prefactor *= -1;
      A1R[dense_fortran(4*i+1,4*i+1, 2*qubits)] = 0;
      A1R[dense_fortran(4*i+2,4*i+2, 2*qubits)] = 0;
      A1R[dense_fortran(4*i+3,4*i+3, 2*qubits)] = 0;
      A1R[dense_fortran(4*i+4,4*i+4, 2*qubits)] = 0;

      A1R[dense_fortran(4*i+1,4*i+3, 2*qubits)] = -1;
      A1R[dense_fortran(4*i+3,4*i+1, 2*qubits)] = 1;
      A1R[dense_fortran(4*i+2,4*i+4, 2*qubits)] = 1;
      A1R[dense_fortran(4*i+4,4*i+2, 2*qubits)] = -1;
    }
  }

  matrix_conjugate_inplace_double(T, A1R, 2*qubits, CblasNoTrans);
  std::vector<std::complex<double> > Tcomplex(2*qubits*2*qubits);
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      Tcomplex[dense_fortran(i+1,j+1,2*qubits)] = T[dense_fortran(i+1,j+1,2*qubits)];
    }
  }
  //add the -iM
  for(int i = 0; i < qubits/2; i++){
    Tcomplex[dense_fortran(4*i+1,4*i+2,2*qubits)] += -1.i;
    Tcomplex[dense_fortran(4*i+2,4*i+1,2*qubits)] +=  1.i;
    Tcomplex[dense_fortran(4*i+3,4*i+4,2*qubits)] += -1.i;
    Tcomplex[dense_fortran(4*i+4,4*i+3,2*qubits)] +=  1.i;
  }

  //add i times the (4+1)th row/col to the (4+2)th row/col
  //this simplifies the extra term \propto iY\otimes (Z-iX) term arising from the Aitken block diagonalization
  //since Z-iX is equal to
  // 1  -i
  // -i -1
  // i times the first row/col ends up deleting the second row/col
  for(int i = 0; i < qubits/2; i++){
    for(int j = 0; j < 2*qubits; j++){
      Tcomplex[dense_fortran(4*i+2,j+1,2*qubits)] += (1.i)*Tcomplex[dense_fortran(4*i+1,j+1,2*qubits)];
      Tcomplex[dense_fortran(j+1,4*i+2,2*qubits)] += (1.i)*Tcomplex[dense_fortran(j+1,4*i+1,2*qubits)];
    }
  }

  for(int i = 0; i < qubits/2; i++){
    double angle = y[i]*M_PI/4. + (A[i])/2;

    while(angle > M_PI/4.){
      angle -= M_PI/2.;
      prefactor = -prefactor;
    }
    while(angle < -M_PI/4.){
      angle += M_PI/2.;
      prefactor = -prefactor;
    }

    prefactor *= (std::cos(angle)*std::cos(angle));
    double t = std::tan(angle);

    //multiply the (4i+1)th row/col by (1-t*t)
    for(int j = 0; j < 2*qubits; j++){
      Tcomplex[dense_fortran(4*i+1,j+1,2*qubits)] *= (1-t*t);
      Tcomplex[dense_fortran(j+1,4*i+1,2*qubits)] *= (1-t*t);
    }

    std::complex<double> val = 2*t;

    Tcomplex[dense_fortran(4*i+1,4*i+3,2*qubits)] += val;
    Tcomplex[dense_fortran(4*i+1,4*i+4,2*qubits)] += -val*1.i;

    Tcomplex[dense_fortran(4*i+3,4*i+1,2*qubits)] += -val;
    Tcomplex[dense_fortran(4*i+4,4*i+1,2*qubits)] += val*1.i;
  }
  prefactor *= std::pow(-1, qubits);

  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(2*qubits, &Tcomplex[0], &pfaffian, "U", "P");

  return prefactor*pfaffian;
}
*/



std::complex<double> inner_prod_internal(int qubits, std::vector<double> A1, DecomposedPassive &p, std::vector<double> A2){
  
  std::vector<double> KT_A1RT(2*qubits*2*qubits);
  std::fill(KT_A1RT.begin(),KT_A1RT.end(), 0.);
  for(int i = 0; i < qubits/2; i++){
    //do 4 columns at a time
    cblas_daxpy(8*qubits, std::cos(A1[i]), &p.R[dense_fortran(1,4*i+1,2*qubits)],1,&KT_A1RT[dense_fortran(1,4*i+1,2*qubits)],1);

    //now do the sin part
    cblas_daxpy(2*qubits, -std::sin(A1[i]), &p.R[dense_fortran(1,4*i+3,2*qubits)],1,&KT_A1RT[dense_fortran(1,4*i+1,2*qubits)],1);
    cblas_daxpy(2*qubits, std::sin(A1[i]), &p.R[dense_fortran(1,4*i+4,2*qubits)],1,&KT_A1RT[dense_fortran(1,4*i+2,2*qubits)],1);
    cblas_daxpy(2*qubits, std::sin(A1[i]), &p.R[dense_fortran(1,4*i+1,2*qubits)],1,&KT_A1RT[dense_fortran(1,4*i+3,2*qubits)],1);
    cblas_daxpy(2*qubits, -std::sin(A1[i]), &p.R[dense_fortran(1,4*i+2,2*qubits)],1,&KT_A1RT[dense_fortran(1,4*i+4,2*qubits)],1);
  }
  
  //left multiply this by C1
  //multiplies alternate rows by cos(theta_j)
  for(int i = 0; i < qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      KT_A1RT[dense_fortran(2*i+1, j+1, 2*qubits)] *= std::cos(p.l[i]/2.);
    }
  }

  //matrix_conjugate_inplace_double(M, KT_A1RT,2*qubits);
  std::vector<double> KT_A1RT2(2*qubits*2*qubits);
  std::fill(KT_A1RT2.begin(), KT_A1RT2.end(), 0.);
  for(int i = 0; i < qubits; i++){
    cblas_daxpy(2*qubits, -1., &KT_A1RT[dense_fortran(1,2*i+1,2*qubits)],1,&KT_A1RT2[dense_fortran(1,2*i+2,2*qubits)],1);
    cblas_daxpy(2*qubits, +1., &KT_A1RT[dense_fortran(1,2*i+2,2*qubits)],1,&KT_A1RT2[dense_fortran(1,2*i+1,2*qubits)],1);
  }
  std::vector<double> M(2*qubits*2*qubits);
  matmul_square_double(CblasNoTrans, CblasTrans, KT_A1RT, KT_A1RT2,  M, 2*qubits);
  
  std::vector<std::complex<double> > G(4*qubits*4*qubits);
  std::fill(G.begin(), G.end(), 0);
  int G_dim = 4*qubits;
  for(int i = 0; i < 2*qubits; i++){
    //this will inly work if your complex number type is actually two adjacent doubles
    //with the first being the real part and the second the imaginary part

    cblas_daxpy(2*qubits, -1., &M[dense_fortran(1,i+1,2*qubits)],1, ((double *)&G[dense_fortran(1,i+1,G_dim)])+1,2);
    
    /*for(int j = 0; j < 2*qubits; j++){
      G[dense_fortran(i+1, j+1, G_dim)] = (-1.i)*M[dense_fortran(i+1,j+1,2*qubits)];
    }
    */
  }

  for(int i = 0; i < qubits; i++){
    G[dense_fortran(2*i+1, 2*i+2, G_dim)] +=std::sin(p.l[i]/2.);
    G[dense_fortran(2*i+2, 2*i+1, G_dim)] +=-std::sin(p.l[i]/2.);
  }
  //const auto t5 = std::chrono::steady_clock::now();
  //bottom right of G
  for(int i = 0; i < qubits/2;i++){
    double angle = (-A1[i] + A2[i])/2;
    G[dense_fortran(2*qubits+4*i+1, 2*qubits+4*i+2, G_dim)] = (-1.i)*std::cos(angle);
    G[dense_fortran(2*qubits+4*i+1, 2*qubits+4*i+3, G_dim)] = std::sin(angle);
    
    G[dense_fortran(2*qubits+4*i+2, 2*qubits+4*i+1, G_dim)] = (1.i)*std::cos(angle);
    G[dense_fortran(2*qubits+4*i+2, 2*qubits+4*i+4, G_dim)] = -std::sin(angle);
    
    G[dense_fortran(2*qubits+4*i+3, 2*qubits+4*i+1, G_dim)] = -std::sin(angle);
    G[dense_fortran(2*qubits+4*i+3, 2*qubits+4*i+4, G_dim)] = (-1.i)*std::cos(angle);

    G[dense_fortran(2*qubits+4*i+4, 2*qubits+4*i+2, G_dim)] = std::sin(angle);
    G[dense_fortran(2*qubits+4*i+4, 2*qubits+4*i+3, G_dim)] = (1.i)*std::cos(angle);
  }
  //top right
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      int sign = 1-2*(j % 2); 
      G[dense_fortran(i+1, 2*qubits+j+1, G_dim)] += KT_A1RT[dense_fortran(i+1,j+1,2*qubits)];
      G[dense_fortran(i+1, 2*qubits+j+1, G_dim)] += (-1.i)*sign*KT_A1RT[dense_fortran(i+1,j+1+sign,2*qubits)];
    }
  }

  //right multiply top right by C2
  for(int i = 0; i < qubits/2; i++){
    double angle = (-A1[i] + A2[i])/2;
    for(int j = 0; j < 2*qubits; j++){      
      G[dense_fortran(j+1, 2*qubits+4*i+1, G_dim)] *= std::cos(angle);
      G[dense_fortran(j+1, 2*qubits+4*i+4, G_dim)] *= std::cos(angle);
    }
  }
  //bottom left
  /*
  for(int i = 0; i < 2*qubits; i++){    
    for(int j = 0; j < 2*qubits; j++){
      G[dense_fortran(2*qubits+i+1, j+1, G_dim)] = -G[dense_fortran(j+1, 2*qubits+i+1, G_dim)];
    }
  }
  */

  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(G_dim, &G[0], &pfaffian, "U", "P");  
  
  return std::pow(-1.i, qubits)*pfaffian*(1-2*((qubits/2)%2));
}

//we compute the inner product of our state KA|0>
//with a computational basis vector |x> where x = y\otimes (1,1)
//i.e. x has a bunch of adjacent pairs of 1s
std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, int y, DecomposedPassive &p, std::vector<double> A2){
  //const auto start = std::chrono::steady_clock::now();
  
  std::vector<double> A1(qubits/2);
  for(int i = 0; i < qubits/2; i++){
    if(((y>>i) & 1) == 1)
    {
      A1[i] = M_PI/2;
    }else{
      A1[i] = 0;
    }
  }
    
  return inner_prod_internal(qubits, A1, p, A2);
}
std::complex<double> inner_prod(int qubits, std::vector<double> A1, std::vector<double> K1, std::complex<double> phase1,std::vector<double> A2, std::vector<double> K2, std::complex<double> phase2)
{
  //first compute K_1^dagger K_2
  std::vector<double> K(2*qubits*2*qubits);
  matmul_square_double(CblasTrans, CblasNoTrans, K1, K2, K, 2*qubits);
  DecomposedPassive p = decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  if(std::abs(p.phase - std::conj(phase1)*phase2) > 1e-10){
    std::cout << "expected phase "<<std::conj(phase1)*phase2<< ", found "<< p.phase << std::endl;  
  }
  return inner_prod_internal(qubits, A1, p, A2);
}

#include "flo.h"
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

  //replace realM with perm @ realM @ perm.T
  matrix_conjugate_inplace_double(realM,perm,2*qubits);

  std::vector<double> T(2*qubits*2*qubits);
  std::fill(T.begin(), T.end(), 0);
  for(int i = 0; i < l.size(); i++){
    double val = tan(l[i]/2.);
    T[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = val;
    T[dense_fortran(4*i+3, 4*i+1, 2*qubits)] = -val;
    T[dense_fortran(4*i+2, 4*i+4, 2*qubits)] = -val;
    T[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = val;
  }
  std::vector<double> scratch(2*qubits*2*qubits);
  std::fill(scratch.begin(), scratch.end(), 0);
  matmul_square_double(x_vec_matrix, perm, scratch, 2*qubits);
  matmul_square_double(CblasNoTrans, CblasTrans, scratch, R, perm,2*qubits);


  matrix_conjugate_inplace_double(T, perm, 2*qubits);

  int w = x_vec_weight; //just a shorthand
  int G_dim = 4*qubits+w;
  std::vector<std::complex<double> > G(G_dim*G_dim);
  std::fill(G.begin(), G.end(), 0);

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
  /* Compute the pfaffian using the lower triangle and the Parlett-Reid
     algorithm */
  info = skpfa(G_dim, &G[0], &pfaffian, "L", "P");

  return prefactor*pfaffian;
}

DecomposedPassive decompose_passive_flo_unitary(std::vector<double> R,int qubits, std::complex<double> phase){
  //we use lapack so we need some different types
  using cdouble = std::complex<double>; 
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


  std::vector<double> V(2*n*2*n);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      V[dense_fortran(2*j+1, 2*i+1, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].real();
      V[dense_fortran(2*j+2, 2*i+2, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].real();
      V[dense_fortran(2*j+1, 2*i+2, 2*n)] =-eigenvectors[dense_fortran(i+1, j+1, n)].imag();
      V[dense_fortran(2*j+2, 2*i+1, 2*n)] = eigenvectors[dense_fortran(i+1, j+1, n)].imag();
    }
  }

  //print_fortran(newR, 2*n);
  //we now compute the log of newR
  //which is block diagonal with 2x2 orthogonal blocks
  //so this is pretty easy
  std::vector<double> lambdas(n);
  double sum = 0;
  for(int i = 0; i < n; i++){
    lambdas[i] = -std::atan2((eigenvalues[i].imag()), (eigenvalues[i].real())); //TODO CHECK THIS!!!!!
    sum += lambdas[i]/2.;
  }
  std::complex<double> new_phase = std::exp(sum*1.i);

  //we might be wrong by a factor of minus 1 in new phase
  //if we are we have to add 2pi to lambda_0
  if(abs(phase - new_phase) > 1e-10){
    lambdas[0] += 2*M_PI;
    new_phase = -new_phase;
  }
  DecomposedPassive p;
  p.phase = new_phase;
  p.l = lambdas;
  p.R = V;


  return p;
}



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
    //this will only work if your complex number type is actually two adjacent doubles
    //with the first being the real part and the second the imaginary part
    cblas_daxpy(2*qubits, -1., &M[dense_fortran(1,i+1,2*qubits)],1, ((double *)&G[dense_fortran(1,i+1,G_dim)])+1,2);    
  }

  for(int i = 0; i < qubits; i++){
    G[dense_fortran(2*i+1, 2*i+2, G_dim)] +=std::sin(p.l[i]/2.);
    G[dense_fortran(2*i+2, 2*i+1, G_dim)] +=-std::sin(p.l[i]/2.);
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
  matmul_square_double(CblasNoTrans, CblasTrans, K2, K1, K, 2*qubits);

  DecomposedPassive p = decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  if(std::abs(p.phase - std::conj(phase1)*phase2) > 1e-10){
    std::cout << "expected phase "<<std::conj(phase1)*phase2<< ", found "<< p.phase << std::endl;  
  }
  return inner_prod_internal(qubits, A1, p, A2);
}



void print_fortran_ignoring_imag(std::vector<std::complex<double> > A, int n){
  for(int i = 1; i <= n; i++){
    for(int j = 1; j <= n; j++){
      std::cout << A[dense_fortran(i,j,n)].real() << " ";
    }
    std::cout << std::endl;
  }
}


std::complex<double> aka_inner_product(int qubits, std::vector<double> V, std::vector<double> A1, DecomposedPassive &p, std::vector<double> A2){
  
  std::vector<std::complex<double> > G(6*qubits*6*qubits);
  std::fill(G.begin(), G.end(), std::complex<double>(0.,0.));
  int G_dim = 6*qubits;
  for(int i = 0; i < 2*qubits; i++){
    G[dense_fortran(i+1, 2*qubits+i+1, G_dim)] = 1;
    G[dense_fortran(2*qubits+i+1, i+1, G_dim)] = -1;

    G[dense_fortran(i+1, 4*qubits+i+1, G_dim)] = -1;
    G[dense_fortran(4*qubits+i+1, i+1, G_dim)] = 1;

    G[dense_fortran(2*qubits+i+1, 4*qubits+i+1, G_dim)] = 1;
    G[dense_fortran(4*qubits+i+1, 2*qubits+i+1, G_dim)] = -1;
  }
  

  std::complex<double> prefactor = 1.;

  //fill in the bottom right which is the generating function of A1 A2

  std::cout << "c++ angles: ";
  for(int i = 0; i < qubits/2; i++){
    double angle = (A1[i] + A2[i])/2.;
    std::cout << angle << ", ";
    prefactor *= (std::cos(angle)*std::cos(angle));
    
    double t = std::tan(angle);
    G[dense_fortran(4*qubits+4*i+1, 4*qubits+4*i+3, G_dim)] = t;
    G[dense_fortran(4*qubits+4*i+2, 4*qubits+4*i+4, G_dim)] = -t;

    G[dense_fortran(4*qubits+4*i+3, 4*qubits+4*i+1, G_dim)] = -t;
    G[dense_fortran(4*qubits+4*i+4, 4*qubits+4*i+2, G_dim)] = t;    
  }
  std::cout << std::endl;
  
  //now we need the middle part which is the generating function of K
  std::vector<double> T(2*qubits*2*qubits, 0.);
  std::fill(T.begin(), T.end(), 0.);
  std::complex<double> passive_phase = 1.;
  for(int i = 0; i < qubits; i++){
    prefactor *= std::cos(p.l[i]/2.);
    passive_phase *= exp(std::complex<double>(0, p.l[i]/2.));
    double t = std::tan(p.l[i]/2.);
    T[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = t;
    T[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -t;
  }
  std::cout << "passive phase = " << passive_phase << std::endl;

  matrix_conjugate_inplace_double(T, p.R, 2*qubits, CblasTrans);

  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      G[dense_fortran(2*qubits+i+1, 2*qubits+j+1, G_dim)] = T[dense_fortran(i+1, j+1, 2*qubits)];
    }
  }

  //now we need the top part left part which is the generating function of P A_1^\dagger V |0><0| V^\dagger A_1 
  std::fill(T.begin(), T.end(), 0.);
  for(int i = 0; i < qubits; i++){
    T[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = 1;
    T[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -1;
  }
  matrix_conjugate_inplace_double(T, V, 2*qubits, CblasNoTrans);
  
  std::vector<double> L1(4*qubits*qubits, 0.);
  for(int i = 0; i < qubits/2; i++){
    L1[dense_fortran(4*i+1, 4*i+1, 2*qubits)] = std::cos(-A1[i]);
    L1[dense_fortran(4*i+2, 4*i+2, 2*qubits)] = std::cos(-A1[i]);
    L1[dense_fortran(4*i+3, 4*i+3, 2*qubits)] = std::cos(-A1[i]);
    L1[dense_fortran(4*i+4, 4*i+4, 2*qubits)] = std::cos(-A1[i]);

    L1[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = std::sin(-A1[i]);
    L1[dense_fortran(4*i+2, 4*i+4, 2*qubits)] =-std::sin(-A1[i]);
    L1[dense_fortran(4*i+3, 4*i+1, 2*qubits)] =-std::sin(-A1[i]);
    L1[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = std::sin(-A1[i]);
  }
  matrix_conjugate_inplace_double(T, L1, 2*qubits, CblasNoTrans);
  
  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      G[dense_fortran(i+1, j+1, G_dim)] = T[dense_fortran(i+1, j+1, 2*qubits)]*std::complex<double>(0.,1.);
    }
  }
  
  //std::cout.width(2);
  //std::cout << std::setprecision(1) << std::showpos << std::fixed;

  //print_fortran(G, G_dim);
  //print_fortran_ignoring_imag(G,G_dim);
  
  std::cout << std::setprecision(6) << std::showpos << std::scientific;
  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(G_dim, &G[0], &pfaffian, "U", "P");

  return std::pow(-1.i, qubits)*pfaffian*prefactor;
}



int aka_to_kak(int qubits, std::vector<double> lambda1, std::vector<double> R, std::complex<double> phase, std::vector<double> lambda2){
  //lambda1 and lambda2 represent antipassive flo unitaries
  //(R, phase) represents a passive flo unitary
  //we seek an alpha K1 A K2 such that
  // L1 K L2 = alpha K1 A K2
  
  //we first multiply L1 K L2 = U in a phase insensitive way
  //and decompose U to find a V such that <0| V^\dagger U V |0> is a complex number of absolute value 1
  //then we KAK decompose U using the symplectic orthogonal block decomposition
  //then we compute <0| V^\dagger K1 A K2 V |0> and <0| V^\dagger L1 K L2 V |0>
  //and compare them so we get the phase correct

  std::vector<double> L1(4*qubits*qubits, 0.);
  std::vector<double> L2(4*qubits*qubits, 0.);
  //turn l1 and l2 into explicit orthogonal matrices
  //according to R = exp(-Lambda\otimes i\sigma_y \otimes \sigma_z)
  //for efficiency one could avoid this (and explicit matrix multiplications) since they are block diagonal 
  for(int i = 0; i < qubits/2; i++){
    L1[dense_fortran(4*i+1, 4*i+1, 2*qubits)] = cos(-lambda1[i]);
    L1[dense_fortran(4*i+2, 4*i+2, 2*qubits)] = cos(-lambda1[i]);
    L1[dense_fortran(4*i+3, 4*i+3, 2*qubits)] = cos(-lambda1[i]);
    L1[dense_fortran(4*i+4, 4*i+4, 2*qubits)] = cos(-lambda1[i]);

    L1[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = sin(-lambda1[i]);
    L1[dense_fortran(4*i+2, 4*i+4, 2*qubits)] =-sin(-lambda1[i]);
    L1[dense_fortran(4*i+3, 4*i+1, 2*qubits)] =-sin(-lambda1[i]);
    L1[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = sin(-lambda1[i]);
  }
  for(int i = 0; i < qubits/2; i++){
    L2[dense_fortran(4*i+1, 4*i+1, 2*qubits)] = cos(-lambda2[i]);
    L2[dense_fortran(4*i+2, 4*i+2, 2*qubits)] = cos(-lambda2[i]);
    L2[dense_fortran(4*i+3, 4*i+3, 2*qubits)] = cos(-lambda2[i]);
    L2[dense_fortran(4*i+4, 4*i+4, 2*qubits)] = cos(-lambda2[i]);

    L2[dense_fortran(4*i+1, 4*i+3, 2*qubits)] = sin(-lambda2[i]);
    L2[dense_fortran(4*i+2, 4*i+4, 2*qubits)] =-sin(-lambda2[i]);
    L2[dense_fortran(4*i+3, 4*i+1, 2*qubits)] =-sin(-lambda2[i]);
    L2[dense_fortran(4*i+4, 4*i+2, 2*qubits)] = sin(-lambda2[i]);
  }

  std::vector<double> U = matmul_square_double(L2, matmul_square_double(R, L1, 2*qubits), 2*qubits);
  std::cout << "U = " << std::endl;
  print_fortran(U, 2*qubits);
  //now we real Schur decompose U
  //this will 2x2 block diagonalise it because it is orthogonal
  //we obtain V from this

  std::vector<double> workopt(1);
  int sdim = 0;
  std::vector<double> schurvectors(4*qubits*qubits);
  std::vector<double> eigenvalues_r(2*qubits);
  std::vector<double> eigenvalues_i(2*qubits);
  std::vector<int> bwork(2*qubits);
  int32_t info;
  int MINUS_1 = -1;
  int n = 2*qubits;
  LAPACK_dgees("V", "N", NULL, &n, &U[0], &n, &sdim,
               &eigenvalues_r[0], &eigenvalues_i[0], &schurvectors[0],
               &n, &workopt[0], &MINUS_1,  NULL, &info);


  int lwork = (int)workopt[0];
  std::vector<double> work(lwork);
  LAPACK_dgees("V", "N", NULL, &n, &U[0], &n, &sdim,
               &eigenvalues_r[0], &eigenvalues_i[0], &schurvectors[0],
               &n, &work[0], &lwork, NULL, &info);

  /*
  for(auto it = U.begin(); it != U.end(); it++){
    if(abs(*it) < 1e-14){
      *it = 0;
    }
    if(abs(*it-1) < 1e-14){
      *it = 1;
    }
  }
  */

  
  
  for(int i = 0; i < qubits; i++){
    if(U[dense_fortran(2*i+1, 2*i+2, 2*qubits)] < 0){
      //swap this row/col pair
      U[dense_fortran(2*i+1, 2*i+2, 2*qubits)] *= -1;
      U[dense_fortran(2*i+2, 2*i+1, 2*qubits)] *= -1;
      //now we need to right multiply a permutation on to schurvectors
      //this swaps columns 2*i+2 and 2*i+1
      cblas_dswap(n, &schurvectors[dense_fortran(1, 2*i+1, 2*qubits)], 1,  &schurvectors[dense_fortran(1, 2*i+2, 2*qubits)], 1);      
    }    
  }
  std::cout << std::endl << std::left << std::setprecision(3) << std::scientific<< std::showpos;
  print_fortran(U, 2*qubits);
  std::cout << "U' = " << std::endl;
  print_fortran(matmul_square_double(schurvectors, matmul_square_double(CblasNoTrans, CblasTrans, U, schurvectors, 2*qubits), 2*qubits), 2*qubits);

  // U_{old} = V U_{new} V^T
  std::vector<double> mu(qubits);
  std::cout << std::endl;
  double sum = 0;
  for(int i = 0; i < qubits; i++){
    mu[i] = atan2(U[dense_fortran(2*i+1, 2*i+2, 2*qubits)], U[dense_fortran(2*i+1, 2*i+1, 2*qubits)]);
    sum += mu[i];
    std::cout << mu[i] << ", ";
  }
  std::cout << std::endl <<"sum = " << sum << std::endl;
  
  std::complex<double> phase2 = exp(std::complex<double>(0.,sum/2.));
  std::cout << "phase2 = " << phase2 << std::endl;
  
  
  //calculate <0| V^T U_{old} V |0>  = <0| U_{new} |0>


  DecomposedPassive p = decompose_passive_flo_unitary(R, qubits, phase);
  std::cout << "phases: " << phase << ", " << p.phase << std::endl;
  std::complex<double> aka_phase = aka_inner_product(qubits, schurvectors, lambda1, p, lambda2);

  std::cout << "aka_phase = " <<aka_phase<<std::endl;
  std::cout << "ratio =  " << phase2/aka_phase << std::endl;
  return 0;
}

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



void left_apply_antipassive(int qubits, std::vector<double> &M, std::vector<double> angle){
  int n = 2*qubits;
  for(int i = 0; i < qubits/2; i++){
    cblas_drot(n, &M[dense_fortran(4*i+1, 1, n)], n, &M[dense_fortran(4*i+3, 1, n)], n, cos(angle[i]), sin(-angle[i]));
    cblas_drot(n, &M[dense_fortran(4*i+2, 1, n)], n, &M[dense_fortran(4*i+4, 1, n)], n, cos(angle[i]), sin(angle[i]));
  }
}

void right_apply_antipassive(int qubits, std::vector<double> &M, std::vector<double> angle){
  int n = 2*qubits;
  for(int i = 0; i < qubits/2; i++){
    cblas_drot(n, &M[dense_fortran(1, 4*i+1, n)], 1, &M[dense_fortran(1, 4*i+3, n)], 1, cos(angle[i]), sin(angle[i]));
    cblas_drot(n, &M[dense_fortran(1, 4*i+2, n)], 1, &M[dense_fortran(1, 4*i+4, n)], 1, cos(angle[i]), sin(-angle[i]));
  }  
}

void conjugate_by_antipassive(int qubits, std::vector<double> &M, std::vector<double> angle){
  int n = 2*qubits;
  for(int i = 0; i < qubits/2; i++){
    cblas_drot(n, &M[dense_fortran(4*i+1, 1, n)], n, &M[dense_fortran(4*i+3, 1, n)], n, cos(angle[i]), sin(-angle[i]));
    cblas_drot(n, &M[dense_fortran(4*i+2, 1, n)], n, &M[dense_fortran(4*i+4, 1, n)], n, cos(angle[i]), sin(angle[i]));
    
    cblas_drot(n, &M[dense_fortran(1, 4*i+1, n)], 1, &M[dense_fortran(1, 4*i+3, n)], 1, cos(angle[i]), sin(-angle[i]));
    cblas_drot(n, &M[dense_fortran(1, 4*i+2, n)], 1, &M[dense_fortran(1, 4*i+4, n)], 1, cos(angle[i]), sin(angle[i]));
  }  
}



std::complex<double> inner_prod_M_P_A(int qubits, std::vector<double> M, DecomposedPassive &p, std::vector<double> A){

  //compute the inner product tr(|M><M| P A)
  //where |M> is a FLO state with covariance matrix M
  //P is passive (type K) and A is anti-passive (type A)
  //let C1 S1 and C2 S2 be the matrices of cosines and sines arising from P and A respectively
  //a fun fact is that the R matrix from p has determinant 1 (never -1) this can be proved from
  //det([[c,s],[-s,c]]) = det(c+is)det(c-is) & det(M) = det(M^T) & c+is is unitary
 
  //we are attempting to compute the pfaffian of the 3x3 block matrix

  //[-iM	R C1 	   -C2      ]
  //[-C1R^T     S1         C1 R^T C2]
  //[C2         -C2RC1     S2       ]

  //after applying block diagonalization, using that M^{-1} = -M = M^T  we obtain
  //[S1         C1 R^T C2]   [C1 R^T]                        [S1         C1 R^T C2]   [C1 R^T iM R C1     -C1 R^T iM C2]
  //[-C2RC1     S2       ] + [-C2   ]   (-iM) [R C1, -C2]  = [-C2RC1     S2       ] - [-C2 iM R C1        C2 iM C2     ]

  //C1 is a block diagonal of blocks of the form
  //[cos(p.l[i]/2.)   0]
  //[0.               1]
  //C2 is a block diagonal of blocks of the form
  //[cos(A[i]/2.)      0     0   0]
  //[0         cos(A[i]/2.)  0   0]
  //[0.             0        1   0]
  //[0              0        0   1]
  
  //A surprising that happens is that we obtain two factors of Pf(-iM) here
  //one from performing a Grassman integral to simplify parity*|M><M|
  //and one from applying the block diagonalization formula
  //this means that we don't need to compute Pf(-iM) properly, we just use Pf(-iM) = Pf(-i V M0 V^T) = Pf(-i M0) det(V)
  //where M0 is the covariance matrix of the vacuum state
  //since V is orthogonal det(V)^2 = 1 so all we need is Pf(-iM0)^2 = (-i)^{2n) = (-1)^n
  
  //lets compute M R first
  //std::vector<double> RT = transpose(p.R, 2*qubits);
  std::vector<double> R2 = p.R;
  std::vector<double> MR = matmul_square_double(CblasNoTrans, CblasTrans, M, R2, 2*qubits);

  std::vector<std::complex<double> > G((4*qubits)*(4*qubits), 0.);

  //add iMR C1 to lower left of G
  //we abuse that a complex double number is just pair of real doubles, real part then imaginary part
  for(int i = 0; i < qubits; i++){
    //add column by column
    cblas_daxpy(2*qubits, cos(p.l[i]/2.), &MR[dense_fortran(1, 2*i+1, 2*qubits)], 1, ((double*)&G[dense_fortran(2*qubits+1, 2*i+1, 4*qubits)])+1, 2);
    cblas_daxpy(2*qubits, 1., &MR[dense_fortran(1, 2*i+2, 2*qubits)], 1, ((double*)&G[dense_fortran(2*qubits+1, 2*i+2, 4*qubits)])+1, 2);
  }

  //add -R C1 to lower left of G
  for(int i = 0; i < qubits; i++){
    cblas_daxpy(2*qubits, -cos(p.l[i]/2.), &R2[dense_fortran(2*i+1, 1, 2*qubits)], 2*qubits, (double*)&G[dense_fortran(2*qubits+1, 2*i+1, 4*qubits)], 2);
    cblas_daxpy(2*qubits, -1, &R2[dense_fortran(2*i+2, 1, 2*qubits)], 2*qubits, (double*)&G[dense_fortran(2*qubits+1, 2*i+2, 4*qubits)], 2);
  }
  
  //add -iM C2 to the lower right of G
  for(int i = 0; i < qubits/2; i++){
    cblas_daxpy(2*qubits, -cos(-A[i]/2.), &M[dense_fortran(1, 4*i+1, 2*qubits)], 1, ((double*)&G[dense_fortran(2*qubits+1, 2*qubits+4*i+1, 4*qubits)])+1, 2);
    cblas_daxpy(2*qubits, -cos(-A[i]/2.), &M[dense_fortran(1, 4*i+2, 2*qubits)], 1, ((double*)&G[dense_fortran(2*qubits+1, 2*qubits+4*i+2, 4*qubits)])+1, 2);

    cblas_daxpy(2*qubits, -1, &M[dense_fortran(1, 4*i+3, 2*qubits)], 1, ((double*)&G[dense_fortran(2*qubits+1, 2*qubits+4*i+3, 4*qubits)])+1, 2);
    cblas_daxpy(2*qubits, -1, &M[dense_fortran(1, 4*i+4, 2*qubits)], 1, ((double*)&G[dense_fortran(2*qubits+1, 2*qubits+4*i+4, 4*qubits)])+1, 2);
  }
  //now we left-multiply the whole lower part of G by C2
  for(int i = 0; i < qubits/2; i++){
    //we do two rows at once
    std::complex<double> c(cos(A[i]/2.), 0.);
    cblas_zscal(4*qubits, &c, &G[dense_fortran(2*qubits+4*i+1, 1, 4*qubits)], 4*qubits);
    cblas_zscal(4*qubits, &c, &G[dense_fortran(2*qubits+4*i+2, 1, 4*qubits)], 4*qubits);
  }
  
  //now add S2 to the lower right part of G
  for(int i = 0; i < qubits/2; i++){
    G[dense_fortran(2*qubits+4*i+1, 2*qubits+4*i+3, 4*qubits)] += sin(A[i]/2.);
    G[dense_fortran(2*qubits+4*i+2, 2*qubits+4*i+4, 4*qubits)] -= sin(A[i]/2.);
    G[dense_fortran(2*qubits+4*i+3, 2*qubits+4*i+1, 4*qubits)] -= sin(A[i]/2.);
    G[dense_fortran(2*qubits+4*i+4, 2*qubits+4*i+2, 4*qubits)] += sin(A[i]/2.);
  }
  
  //now top left of G
  std::vector<double> RTMR = matmul_square_double(CblasNoTrans, CblasNoTrans, R2, MR, 2*qubits);

  for(int i = 0; i < qubits; i++){
    //we left multiply RTMR by c1
    //multiplies even rows by cos
    cblas_dscal(2*qubits, cos(p.l[i]/2), &RTMR[dense_fortran(2*i+1, 1, 2*qubits)], 2*qubits); //rows
    //cblas_dscal(2*qubits, cos(p.l[i]/2), &RTMR[dense_fortran(1, 2*i+1, 2*qubits)], 1); //cols    
  }
  //add -C1 RT iM R to top left of G while simultaniously right multiplying by C1
  for(int i = 0; i < qubits; i++){
    cblas_daxpy(2*qubits, -cos(p.l[i]/2), &RTMR[dense_fortran(1, 2*i+1, 2*qubits)], 1, ((double*)&G[dense_fortran(1, 2*i+1, 4*qubits)])+1, 2);
    cblas_daxpy(2*qubits, -1, &RTMR[dense_fortran(1, 2*i+2, 2*qubits)], 1, ((double*)&G[dense_fortran(1, 2*i+2, 4*qubits)])+1, 2);
  }

  //now add S1 to top right of G
  for(int i = 0; i < qubits; i++){
    G[dense_fortran(2*i+1, 2*i+2, 4*qubits)] += sin(p.l[i]/2);
    G[dense_fortran(2*i+2, 2*i+1, 4*qubits)] -= sin(p.l[i]/2);
  }

  std::complex<double> pfaffian=0;
  int info = skpfa_z(4*qubits, &G[0], &pfaffian, "L", "P");
  return pfaffian*(1-2*(qubits%2));

}

//we compute the inner product of our state KA|0>
//with a computational basis vector |x> where x = y\otimes (1,1)
//i.e. x has a bunch of adjacent pairs of 1s
std::complex<double> cb_inner_prod_adjacent_qubits(int qubits, int y, DecomposedPassive &p, std::vector<double> A){
  std::vector<double> M(2*qubits*2*qubits, 0.);
  for(int i = 0; i < qubits; i++){
    M[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = 1;
    M[dense_fortran(2*i+2, 2*i+1, 2*qubits)] =-1;
  }

  for(int i = 0; i < qubits/2; i++){
    if(((y >> i) & 1) == 1)
    {
      A[i] -=  M_PI/2.;
      M[dense_fortran(4*i+1, 4*i+2, 2*qubits)] = -1;
      M[dense_fortran(4*i+2, 4*i+1, 2*qubits)] = 1;
      M[dense_fortran(4*i+3, 4*i+4, 2*qubits)] = -1;
      M[dense_fortran(4*i+4, 4*i+3, 2*qubits)] = 1;      
    }
  }

  return inner_prod_M_P_A(qubits, M, p, A);   
}
std::complex<double> inner_prod(int qubits, std::vector<double> A1, PassiveFLO K1 ,std::vector<double> A2, PassiveFLO K2)
{
  //first compute K_1^dagger K_2

  PassiveFLO K = PassiveFLO::multiply(CblasTrans, CblasNoTrans, K1, K2);
  //std::vector<double> K(2*qubits*2*qubits);
  //matmul_square_double(CblasNoTrans, CblasTrans, K2, K1, K, 2*qubits);

  DecomposedPassive p = K.decompose(); //decompose_passive_flo_unitary(K, qubits, std::conj(phase1)*phase2);
  if(std::abs(p.phase - std::conj(*K1.phase)*(*K2.phase)) > 1e-10){
    std::cout << "expected phase "<<std::conj(*K1.phase)*(*K2.phase)<< ", found "<< p.phase << std::endl;  
  }

  
  std::vector<double> M(2*qubits*2*qubits, 0.);
  for(int i = 0; i < qubits; i++){
    M[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = 1;
    M[dense_fortran(2*i+2, 2*i+1, 2*qubits)] =-1;
  }
  
  
  std::vector<double> A(qubits/2, 0.);
  for(int i = 0; i < qubits/2; i++){
    A[i] = -A1[i] + A2[i];
    A1[i] *= -1;
  }

  conjugate_by_antipassive(qubits, M, A1);
  DecomposedPassive KDecomp = K.decompose();
  return inner_prod_M_P_A(qubits, M, KDecomp, A);
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


  for(int i = 0; i < qubits/2; i++){
    double angle = (A1[i] + A2[i])/2.;

    prefactor *= (std::cos(angle)*std::cos(angle));
    
    double t = std::tan(angle);
    G[dense_fortran(4*qubits+4*i+1, 4*qubits+4*i+3, G_dim)] = t;
    G[dense_fortran(4*qubits+4*i+2, 4*qubits+4*i+4, G_dim)] = -t;

    G[dense_fortran(4*qubits+4*i+3, 4*qubits+4*i+1, G_dim)] = -t;
    G[dense_fortran(4*qubits+4*i+4, 4*qubits+4*i+2, G_dim)] = t;    
  }

  
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


  matrix_conjugate_inplace_double(T, p.R, 2*qubits, CblasNoTrans);

  for(int i = 0; i < 2*qubits; i++){
    for(int j = 0; j < 2*qubits; j++){
      G[dense_fortran(2*qubits+i+1, 2*qubits+j+1, G_dim)] = T[dense_fortran(i+1, j+1, 2*qubits)];
    }
  }

  //now we need the top part left part which is the generating function of P A_1^\dagger V |0><0| V^\dagger A_1 
  std::fill(T.begin(), T.end(), 0.);
  for(int i = 0; i < qubits; i++){
    T[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = -1;
    T[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = 1;
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
      G[dense_fortran(i+1, j+1, G_dim)] = T[dense_fortran(i+1, j+1, 2*qubits)]*std::complex<double>(0.,-1.);
    }
  }

  //schur decompose V because we care about its determinant (it might be minus 1)
  //because our answer will be multiplied by Pf(-i\tilde{M}) = det(V) Pf(-iM) = det(V) (-i)^n
  std::vector<double> workopt(1);
  int sdim = 0;
  std::vector<double> eigenvalues_r(2*qubits);
  std::vector<double> eigenvalues_i(2*qubits);
  std::vector<int> bwork(2*qubits);
  int32_t info_schur;
  int MINUS_1 = -1;
  int n = 2*qubits;
  LAPACK_dgees("N", "N", NULL, &n, &V[0], &n, &sdim,
               &eigenvalues_r[0], &eigenvalues_i[0], NULL,
               &n, &workopt[0], &MINUS_1,  NULL, &info_schur);


  int lwork = (int)workopt[0];
  std::vector<double> work(lwork);
  LAPACK_dgees("N", "N", NULL, &n, &V[0], &n, &sdim,
               &eigenvalues_r[0], &eigenvalues_i[0], NULL,
               &n, &work[0], &lwork, NULL, &info_schur);

  std::complex<double> det = 1.;
  for(int i = 0; i < 2*qubits; i++){
    det *= std::complex<double>(eigenvalues_r[i], eigenvalues_i[i]);
  }
  //std::cout << "V determinant = " << det << std::endl;
  
  //std::cout.width(2);
  //std::cout << std::setprecision(1) << std::showpos << std::fixed;

  //print_fortran(G, G_dim);
  //print_fortran_ignoring_imag(G,G_dim);
  
  
  std::complex<double> pfaffian=0;
  int info;
  info = skpfa_z(G_dim, &G[0], &pfaffian, "U", "P");

  return std::pow(-1.i, qubits)*pfaffian*prefactor*det;
}

std::vector<double> transpose(std::vector<double> R, int n){
  std::vector<double> V(n*n,0.);
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      V[dense_fortran(i+1, j+1, n)] = R[dense_fortran(j+1, i+1, n)];
    }
  }
  return V;
}


std::tuple<std::vector<double>,std::vector<double>,std::vector<double> > KAK_decompose(std::vector<double> R, int qubits){
  std::vector<double> Q1(2*qubits*2*qubits, 0.);
  std::vector<double> Q2(2*qubits*2*qubits, 0.);
  std::vector<double> A = symplectic_orthogonal_factorize(qubits, R, Q1, Q2);
  //std::cout << "A = "<< std::endl;
  //print_fortran(A, 2*qubits);
  //std::cout << std::endl;
  //A has the form
  //[1  0  0  0]
  //[0  c  0  s]
  //[0  0  1  0]
  //[0 -s  0  1]
  //but we want the form
  //[ c  0  s  0]
  //[ 0  c  0 -s]
  //[-s  0  c  0]
  //[ 0  s  0  c]

  Q1 = transpose(Q1, 2*qubits);
  Q2 = transpose(Q2, 2*qubits);

  std::vector<double> lambda(qubits/2, 0.);

  for(int i = 0; i < qubits/2; i++){
    double theta = atan2(A[dense_fortran(4*i+2, 4*i+4, 2*qubits)], A[dense_fortran(4*i+2, 4*i+2, 2*qubits)]);
    lambda[i] = -theta/2;
    //std::cout << i << " " << theta << " " << cos(theta/2) << " " << sin(theta/2) << std::endl;
    //now we right-multiply the symplectic-orthogonal transformation 
    //[ c  0  s  0]
    //[ 0  c  0  s]
    //[-s  0  c  0]
    //[ 0 -s  0  c]
    //onto Q1, where c and s are cos and sin of theta/2
    cblas_drot(2*qubits, &Q1[dense_fortran(1, 4*i+1, 2*qubits)], 1, &Q1[dense_fortran(1, 4*i+3, 2*qubits)], 1, cos(theta/2), -sin(theta/2));
    cblas_drot(2*qubits, &Q1[dense_fortran(1, 4*i+2, 2*qubits)], 1, &Q1[dense_fortran(1, 4*i+4, 2*qubits)], 1, cos(theta/2), -sin(theta/2));    
  }

  return std::tuple<std::vector<double>, std::vector<double>, std::vector<double> >(Q1, lambda, Q2);
}



std::tuple<std::complex<double>, PassiveFLO, std::vector<double>, PassiveFLO> aka_to_kak(int qubits, std::vector<double> lambda1, PassiveFLO K, std::vector<double> lambda2){
  //lambda1 and lambda2 represent antipassive flo unitaries
  //(R, phase) represents a passive flo unitary
  //we seek an alpha K1 A K2 such that
  // L1 K L2 = alpha K1 A K2
  
  //we first multiply L1 K L2 = U in a phase insensitive way
  //and decompose U to find a V such that <0| V^\dagger U V |0> is a complex number of absolute value 1
  //then we KAK decompose U using the symplectic orthogonal block decomposition
  //then we compute <0| V^\dagger K1 A K2 V |0> and <0| V^\dagger L1 K L2 V |0>
  //and compare them so we get the phase correct
  
  std::vector<double> U = K.R;
  left_apply_antipassive(qubits, U, lambda2);
  right_apply_antipassive(qubits, U, lambda1);
  std::vector<double> Ucpy = U; //we need a copy of U for later and dgees will overwrite it
  
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

  // U_{old} = V U_{new} V^T
  //calculate <0| V^T U_{old} V |0>  = <0| U_{new} |0>
  double sum = 0;
  for(int i = 0; i < qubits; i++){
    sum += atan2(U[dense_fortran(2*i+1, 2*i+2, 2*qubits)], U[dense_fortran(2*i+1, 2*i+1, 2*qubits)]);
  }
  std::complex<double> phase2 = exp(-std::complex<double>(0.,sum/2.));

  //now compute <0| V^T A_1 K A_2 V |0>
  std::vector<double> Rcpy = K.R;
  DecomposedPassive p = K.decompose(); //decompose_passive_flo_unitary(R, qubits, phase);

  std::vector<double> M(4*qubits*qubits, 0.);
  for(int i = 0; i < qubits; i++){
    M[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = 1;
    M[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -1;
  }
  // M -> (VL)^T M (VL)
  matrix_conjugate_inplace_double(M, schurvectors, 2*qubits, CblasNoTrans);
  //matrix_conjugate_inplace_double(M, L, 2*qubits, CblasNoTrans);
  conjugate_by_antipassive(qubits, M, lambda1);

  std::vector<double> lambda(qubits/2, 0);
  for(int i = 0; i < qubits/2; i++){
    lambda[i] = lambda1[i] + lambda2[i];
  }

  //compute <0| A1 K A2 |0>
  std::complex<double> M_P_A = inner_prod_M_P_A(qubits, M, p, lambda);
  

  std::tuple<std::vector<double>,std::vector<double>,std::vector<double> > t = KAK_decompose(Ucpy, qubits);

  
  //std::vector<double> newA(2*qubits*2*qubits, 0.);
  std::vector<double> new_a_lambda(qubits/2);
  for(int i = 0; i < qubits/2; i++){
    double angle = std::get<1>(t)[i];
    new_a_lambda[i] = -1*std::get<1>(t)[i];
  }
  
  PassiveFLO K1(qubits, std::get<0>(t));  
  PassiveFLO K2(qubits, std::get<2>(t));

  //K1 A K2 == U as orthogonal matrices
  //so K2 A K1 == U as FLO unitaries
  
  //now we compute tr[V|0><0|V^\dagger K2 A K1] = tr[K1 V|0><0|(K1 V)^\dagger K1 K2 A]
  std::fill(M.begin(), M.end(), 0.);
  for(int i = 0; i < qubits; i++){
    M[dense_fortran(2*i+1, 2*i+2, 2*qubits)] = 1;
    M[dense_fortran(2*i+2, 2*i+1, 2*qubits)] = -1;
  }

  matrix_conjugate_inplace_double(M, schurvectors, 2*qubits, CblasNoTrans);
  matrix_conjugate_inplace_double(M, K1.R, 2*qubits, CblasTrans);

  
  DecomposedPassive k1_decomp = K1.decompose(); //= decompose_passive_flo_unitary(K1, qubits, 0.); 
  
  DecomposedPassive k2k1_decomp = PassiveFLO::multiply(CblasNoTrans, CblasNoTrans, K1, K2).decompose();
  //decompose_passive_flo_unitary(matmul_square_double(CblasNoTrans, CblasNoTrans, K2, K1, 2*qubits), qubits, 0.);


  std::complex<double> M_P_A2 = inner_prod_M_P_A(qubits, M, k2k1_decomp, new_a_lambda);


  if(abs(M_P_A - M_P_A2) > abs(M_P_A + M_P_A2)){
    k1_decomp.phase *= -1;
    M_P_A2 *= -1;
  }

  K1.phase = k1_decomp.phase;
  K2.phase = k2k1_decomp.phase/k1_decomp.phase;
  
  //we return innerproduct, (K1, K1phase), (K2, K2phase), new_a_lambda
  //enough information to completely reproduce the KAK decompostion of U
  
  return std::tuple<std::complex<double>, PassiveFLO, std::vector<double>, PassiveFLO >(M_P_A2, K1, new_a_lambda, K2);
}

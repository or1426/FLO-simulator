#include "flo.h"
#include <complex>
using namespace std::complex_literals;
#include <ceres/ceres.h>

#include <iostream>
#include <numeric>
#include <random>
	

#define COUNT 4
#define QUBITS (4*COUNT)
#define VECS ((1<<COUNT) -1)
#define ALPHAS_SIZE (VECS*QUBITS*QUBITS)
#define LAMBDAS_SIZE (VECS*QUBITS/2)

struct CostFunctor {
  bool operator()(const double* const as, const double* const alphas, const double* const lambdas, double* residual) const {
    // we first want to turn our alphas into explicit orthogonal matrices & phases
    using cdouble = std::complex<double>;
    std::vector<std::vector<cdouble> > A_vecs(VECS);
    std::vector<std::vector<double> > lambda_vecs(VECS);
    
    int alphas_idx = 0;
    int lambdas_idx = 0;
    
    for(int idx = 0; idx < VECS; idx++){
      //each alpha is a 2n*2n dimensional skew symmetric matrix
      //which commutes with (I \otimes iY)
      //and is therefore of the form (A\otimes I) + (B\otimes iY)
      //where A is skew symmetric & B is symmetric

      A_vecs[idx].resize(2*QUBITS*2*QUBITS);
      std::fill(A_vecs[idx].begin(), A_vecs[idx].end(), 0.);
      for(int i = 0; i < QUBITS; i++){
        //skew part
        for(int j = 0; j < i; j++){
          A_vecs[idx][dense_fortran(i+1,j+1, QUBITS)] += -alphas[alphas_idx]*(1.i);
          A_vecs[idx][dense_fortran(j+1,i+1, QUBITS)] += alphas[alphas_idx]*(1.i);
          /*
            alpha_vecs[idx][dense_fortran(2*i+1,2*j+1, 2*QUBITS)] = alphas[alphas_idx];
            alpha_vecs[idx][dense_fortran(2*i+2,2*j+2, 2*QUBITS)] = alphas[alphas_idx];

            alpha_vecs[idx][dense_fortran(2*j+1,2*i+1, 2*QUBITS)] = -alphas[alphas_idx];
            alpha_vecs[idx][dense_fortran(2*j+1, 2*i+1, 2*QUBITS)] = -alphas[alphas_idx];
          */
          alphas_idx += 1;
        }
        A_vecs[idx][dense_fortran(i+1,i+1, QUBITS)] += alphas[alphas_idx];
        alphas_idx += 1;
        //symmetric part
        for(int j = i+1; j < QUBITS; j++){
          A_vecs[idx][dense_fortran(i+1,j+1, QUBITS)] += alphas[alphas_idx];
          A_vecs[idx][dense_fortran(j+1,i+1, QUBITS)] += alphas[alphas_idx];
          /*
            alpha_vecs[idx][dense_fortran(2*i+1,2*j+2, 2*QUBITS)] = alphas[alphas_idx];
            alpha_vecs[idx][dense_fortran(2*i+2,2*j+1, 2*QUBITS)] = -alphas[alphas_idx];

            alpha_vecs[idx][dense_fortran(2*j+1,2*i+2, 2*QUBITS)] = alphas[alphas_idx];
            alpha_vecs[idx][dense_fortran(2*j+2,2*i+1, 2*QUBITS)] = -alphas[alphas_idx];
          */
          alphas_idx += 1;
        }
      }

      lambda_vecs[idx].resize(QUBITS/2);
      for(int i = 0; i < QUBITS/2; i++){
        lambda_vecs[idx][i] = lambdas[lambdas_idx];
        lambdas_idx += 1;
      }
    }

    //now we take each alpha & compute the R matrix & phase
    //we do this by diagonalizing the Hermitian A matrices we made

    	  A2[dense_fortran(2*i+1,2*j+1, 2*QUBITS)] = -cimag(A_vecs[idx][dense_fortran(i+1,j+1, QUBITS)]);
	  A2[dense_fortran(2*i+2,2*j+2, 2*QUBITS)] = -cimag(A_vecs[idx][dense_fortran(i+1,j+1, QUBITS)]);

	  A2[dense_fortran(2*i+1,2*j+2, 2*QUBITS)] = creal(A_vecs[idx][dense_fortran(i+1,j+1, QUBITS)]);
	  A2[dense_fortran(2*i+2,2*j+1, 2*QUBITS)] =-creal(A_vecs[idx][dense_fortran(i+1,j+1, QUBITS)]);
	}
	}*/
      
      
      int n = QUBITS;
      int lda = QUBITS;
      int lwork = -1;
      int info = 0;
      std::vector<double> w(QUBITS);
      std::vector<double> rwork(3*n - 2);
      cdouble wkopt;
      LAPACK_zheev( "Vectors", "Lower", &n, &A_vecs[idx][0], &lda, &w[0], &wkopt, &lwork, &rwork[0], &info );
      lwork = (int)wkopt.real();
      std::vector<cdouble> work(lwork);
      //work = (dcomplex*)malloc( lwork*sizeof(dcomplex) );
      LAPACK_zheev( "Vectors", "Lower", &n, &A_vecs[idx][0], &lda, &w[0], &work[0], &lwork, &rwork[0], &info);
      //std::cout << "eigenvals ";
      //for(int i = 0; i < QUBITS; i++){
      //std::cout << w[i] << " ";
      //}
      //std::cout << std::endl;
      //A_vecs[idx] now contains the eigenvectors
      //and w the eigenvalues
      /*
      std::vector<double> newR(2*n*2*n);
      
      
      for(int i = 0; i < n; i++){
	newR[dense_fortran(2*i+1, 2*i+1, 2*n)] = creal(w[i]);
	newR[dense_fortran(2*i+2, 2*i+2, 2*n)] = creal(w[i]);
	
	newR[dense_fortran(2*i+1, 2*i+2, 2*n)] = cimag(w[i]);
	newR[dense_fortran(2*i+2, 2*i+1, 2*n)] =-cimag(w[i]);
      }
      */
      std::vector<double> V(2*n*2*n);
      for(int i = 0; i < n; i++){
	for(int j = 0; j < n; j++){
	  V[dense_fortran(2*j+1, 2*i+1, 2*n)] = A_vecs[idx][dense_fortran(i+1, j+1, n)].real();
	  V[dense_fortran(2*j+2, 2*i+2, 2*n)] = A_vecs[idx][dense_fortran(i+1, j+1, n)].real();
	  V[dense_fortran(2*j+1, 2*i+2, 2*n)] =-A_vecs[idx][dense_fortran(i+1, j+1, n)].imag();
	  V[dense_fortran(2*j+2, 2*i+1, 2*n)] = A_vecs[idx][dense_fortran(i+1, j+1, n)].imag();
	}
      }
      //std::cout <<std::endl;
      //print_fortran(A2, 2*QUBITS);
      //std::cout <<std::endl;
      /*
      matrix_conjugate_inplace_double(A2, V, 2*QUBITS);
      for(int i = 0; i < 2*QUBITS; i++){
	for(int j = 0; j < 2*QUBITS; j++){
	  if(std::abs(A2[dense_fortran(i+1,j+1, 2*QUBITS)]) < 1e-14){
	    A2[dense_fortran(i+1,j+1, 2*QUBITS)] = 0;
	  }
	}
	}*/
      //print_fortran(A2, 2*QUBITS);

      std::vector<double> R(2*QUBITS*2*QUBITS);
      std::fill(R.begin(), R.end(), 0);
      double total_angle = 0;
      for(int i = 0; i < QUBITS; i++){
	total_angle += -w[i]/2;
	R[dense_fortran(2*i+1, 2*i+1, 2*QUBITS)] = std::cos(w[i]);
	R[dense_fortran(2*i+2, 2*i+2, 2*QUBITS)] = std::cos(w[i]);
	
	R[dense_fortran(2*i+1, 2*i+2, 2*QUBITS)] =-std::sin(w[i]);
	R[dense_fortran(2*i+2, 2*i+1, 2*QUBITS)] =std::sin(w[i]);

	w[i] = -w[i];
      }

      matrix_conjugate_inplace_double(R, V, 2*QUBITS, CblasTrans);

      DecomposedPassive p;
      p.phase = std::exp((1.i)*total_angle);
      p.l = w;
      p.R = V;
      
      decomposedPassiveVec.push_back(p);
      rVec.push_back(R);
    }


    residual[0] = 0.;
    //as[0] = 1;
    //as[1] = 0;
    //add sum_jk a_j^* a_k <psi_j|psi_k> to residual
    for(int i = 0; i < VECS; i++){
      residual[0] += as[2*i]*as[2*i] + as[2*i+1]*as[2*i+1];      
      for(int j = 0; j < i; j++){
	std::complex<double> prod = inner_prod(QUBITS, lambda_vecs[i], rVec[i], decomposedPassiveVec[i].phase, lambda_vecs[j], rVec[j], decomposedPassiveVec[j].phase);
        residual[0] += 2.*((as[2*i] - as[2*i+1]*(1.i))*(as[2*j] + as[2*j+1]*(1.i))*prod).real();
      }
      for(int y = 0; y < (1<<COUNT); y++){
	int x = 0;
	for(int k = 0; k < COUNT; k++){
	  if(((y>>k) & 1) == 1){
	    x ^= (1<<(2*k));
	    x ^= (1<<(2*k+1));
	  }
	}
	
	std::complex<double> prod = cb_inner_prod_adjacent_qubits(QUBITS, x, decomposedPassiveVec[i],  lambda_vecs[i]);
	residual[0] -= 2*std::pow(1/sqrt(2.), COUNT)*((as[2*i] + as[2*i+1]*(1.i))*prod).real();
	//residual[0] -= 2*std::pow(2., -COUNT/2.)*prod.real();
      }
    }

    residual[0] += 1;
    residual[0] = std::sqrt(2*residual[0]);
    return true;
  }
};


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  //double x = 0.5;
  //const double initial_x = x;
  // Build the problem.
  ceres::Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // numeric differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function =
    new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 1, 2*VECS, ALPHAS_SIZE, LAMBDAS_SIZE>(
      new CostFunctor);
  int val = 1000;
  if(argc >1){
    val = atoi(argv[1]);
  }
  std::mt19937 gen(val);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  
  std::vector<double> as(2*VECS);
  std::fill(as.begin(), as.end(),std::pow(1/2., COUNT/2.));

  std::vector<double> alphas(ALPHAS_SIZE);
  for(int i = 0; i < ALPHAS_SIZE; i++){
    alphas[i] = dis(gen);
  }

  std::vector<double> lambdas(LAMBDAS_SIZE);
  for(int i = 0; i < LAMBDAS_SIZE; i++){
    lambdas[i] = dis(gen);
  }
  
  
  problem.AddResidualBlock(cost_function, nullptr, &as[0], &alphas[0], &lambdas[0]);
  // Run the solver!
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 1000;
  options.minimizer_type = ceres::LINE_SEARCH;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.num_threads = 1;
  options.use_explicit_schur_complement = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;
  std::cout << "seed was " << val << std::endl;
  /*
  std::cout << "a: " << as[0] << " " << as[1] << " " << as[0]*as[0]+as[1]*as[1]<< std::endl;
  for(int i = 0; i < ALPHAS_SIZE; i++){
    std::cout << alphas[i] << " ";
  }
  std::cout << std::endl;
  for(int i = 0; i < LAMBDAS_SIZE; i++){
    std::cout << lambdas[i] << " ";
    }*/
  //std::cout << std::endl;
  //CostFunctorInstrumented f;
  //std::vector<double> r(1);
  //f(&as[0], &alphas[0], &lambdas[0], &r[0]);
  
  //std::cout << "x : " << initial_x << " -> " << x << "\n";
				 
  /*
  struct CostFunctor f;
  
  std::vector<double> A(QUBITS*QUBITS);
  std::vector<double> l(QUBITS/2);

  //std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(1000); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  
  for(int i = 0; i < QUBITS*QUBITS; i++){
    A[i] = dis(gen);
  }
  for(int i = 0; i < QUBITS/2; i++){
    l[i] = dis(gen);
  }
  std::vector<double> alphas(1);
  
  alphas[0] = dis(gen);
  std::vector<double> res(1);
  f(&alphas[0], &A[0], &l[0], &res[0]);
  */
  
  return 0;
}

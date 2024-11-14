# FLO-simulator
Implementation of algorithms I developed for https://arxiv.org/abs/2307.12702

The state of this code is very much "works on my machine". Feel free to contact me if you run into issues.

## Dependencies
See the included makefile

### Standard dependencies
lapack, cblas, blas, gfortran

### pfapack 
Available at https://arxiv.org/abs/1102.3440. Compile it and copy the relevant files over. We need fortran.h, fortran_pfapack.h, libcpfapack.a, libpfapack.a, pfapack.h..

In ourder to run the python tests you will also need the python implementation of pfapack, see https://github.com/basnijholt/pfapack

### nlohmann's json.hpp
Available at https://github.com/nlohmann/json. Only needed if you want to use the python file to make test data to test the c++ code.

### Google's Ceres optimization library

Available at http://ceres-solver.org/. Only needed if you want to try to optimize FLO fidelities using the ceres-optim.cc file

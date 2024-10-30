CC = g++
CFLAGS = -O2 -I/usr/include/x86_64-linux-gnu  
CERESINCLUDES = -I../cereslib/include -I../cereslib/include/ceres/internal/miniglog
LFLAGS = ./libcpfapack.a ./libpfapack.a -llapack -lcblas -lblas -lgfortran
CERESLIBS = ../cereslib/lib/libceres.a

all: tests.out ceres-optim.out

tests:
	$(CC) $(CFLAGS) wrappers.cc so-factorization.cc ff.cc tests.cc -o tests.out $(LFLAGS)

ceres-optim: ceres-optim.cc
	$(CC) $(CFLAGS) $(CERESINCLUDES) wrappers.cc ff.cc ceres-optim.cc -o ceres-optim.out $(LFLAGS) $(CERESLIBS)

clean:
	rm -f *.out

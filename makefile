CC = g++
CFLAGS = -g -I/usr/include/x86_64-linux-gnu  
CERESINCLUDES = -I../cereslib/include -I../cereslib/include/ceres/internal/miniglog
LFLAGS = ./libcpfapack.a ./libpfapack.a -llapack -lcblas -lblas -lgfortran
CERESLIBS = ../cereslib/lib/libceres.a

all: tests.out ceres-optim.out

tests:
	$(CC) $(CFLAGS) wrappers.cc so-factorization.cc passive.cc flo.cc tests.cc -o tests.out $(LFLAGS)
test-state:
	$(CC) $(CFLAGS) wrappers.cc so-factorization.cc passive.cc flo.cc test-flo-state.cc  -o test-flo-state.out $(LFLAGS)
	$(CC) $(CFLAGS) wrappers.cc so-factorization.cc passive.cc flo.cc test-state-dense.cc -o test-state-dense.out $(LFLAGS)	

pt:
	$(CC) $(CFLAGS) wrappers.cc so-factorization.cc passive.cc flo.cc passive-test.cc -o passive-test.out $(LFLAGS)

ceres-optim: ceres-optim.cc
	$(CC) $(CFLAGS) $(CERESINCLUDES) wrappers.cc flo.cc ceres-optim.cc -o ceres-optim.out $(LFLAGS) $(CERESLIBS)

clean:
	rm -f *.out

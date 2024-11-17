#! /usr/bin/env python3
import numpy as np
from pfapack import pfaffian as pf
from numpy.random import default_rng
from scipy import linalg
import sys 
import json



def compute_majs(qubits):
    pI = np.eye(2, dtype=complex)
    pX = np.array([[0,1],[1,0]], dtype=complex)
    pY = np.array([[0,-1j],[1j,0]], dtype=complex)
    pZ = np.array([[1,0],[0,-1]], dtype=complex)
    
    majs = []
    
    
    for i in range(qubits):
        even = pX
        odd = pY
        
        for j in range(i):
            even = np.kron(pZ, even,)
            odd = np.kron(pZ,odd)
        for j in range(i+1, qubits):
            even = np.kron(even, pI)
            odd = np.kron(odd,pI)
        majs.append(even)
        majs.append(odd)
    return majs


def make_symplectic_orthogonal_decompostion_test(seed=1000, count = 2, qubits = 4):
    rng = default_rng(seed)
    
    for _ in range(count):
        A = rng.random((2*qubits,2*qubits), dtype=np.float64)
        A = (A - A.T)/2.
        R = linalg.expm(A)
        obj = {"type": "symplectic-orthogonal-decomposition", 
               "qubits": qubits,
               "R": list(R.T.reshape(2*qubits*2*qubits))
               }
        print(json.dumps(obj))

def make_aka_kak_test(seed=1000, count = 2, qubits = 4):
    rng = default_rng(seed)
    majs = compute_majs(qubits)
    for _ in range(count):
        A = rng.random((qubits,qubits), dtype=np.float64)
        B = rng.random((qubits,qubits), dtype=np.float64)
        A = (A - A.T)/2
        B = (B + B.T)/2
        alpha = (np.kron(A, np.eye(2,dtype=np.float64)) + np.kron(B, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R = linalg.expm(-alpha)
        phase = np.exp(-(1.j/4.)*(alpha @ np.kron(np.eye(qubits,dtype=complex), np.array([[0,1],[-1,0]],dtype=complex))).trace())
        
        l1 = list(rng.random(qubits//2, dtype=np.float64))
        l2 = list(rng.random(qubits//2, dtype=np.float64))
        #print(l1, file=sys.stderr)
        #print(l2, file=sys.stderr)
        #l1[0] = np.pi
        
        majs = compute_majs(qubits)
        exponent_alpha = np.zeros_like(majs[0])
        exponent_l1 = np.zeros_like(majs[0])
        exponent_l2 = np.zeros_like(majs[0])

        for i in range(qubits//2):
            exponent_l1 += l1[i] * (majs[4*i]@majs[4*i+2] - majs[4*i+1]@majs[4*i+3])/2.
            exponent_l2 += l2[i] * (majs[4*i]@majs[4*i+2] - majs[4*i+1]@majs[4*i+3])/2.
        
        for i in range(2*qubits):            
            for j in range(2*qubits):
                exponent_alpha += alpha[i][j]*majs[i]@majs[j]/4.
        #print(np.allclose(linalg.expm(exponent_l1), -), file=sys.stderr)
        #print(np.allclose(linalg.expm(exponent_l1), majs[0]@majs[1]@majs[2]@majs[3]), file=sys.stderr)
        #print("max error:", np.max(np.abs(linalg.expm(exponent_l1) +np.kron(np.kron(np.array([[1,0],[0,-1]]),np.array([[1,0],[0,-1]])), np.eye(4)))), file=sys.stderr)
        
        mat = linalg.expm(exponent_l1) @ linalg.expm(exponent_alpha) @ linalg.expm(exponent_l2)
        
        #print("python 00 phase: ", mat[0,0], file=sys.stderr)
        #print("python 11 phase: ", mat[2**qubits-1,2**qubits-1], file=sys.stderr)
        l1A = np.zeros_like(R)
        l2A = np.zeros_like(R)

        for i in range(qubits//2):
            l1A[4*i, 4*i+2] = l1[i]
            l1A[4*i+1, 4*i+3] = -l1[i]
            l1A[4*i+2, 4*i] = -l1[i]
            l1A[4*i+3, 4*i+1] = l1[i]

            l2A[4*i, 4*i+2] = l2[i]
            l2A[4*i+1, 4*i+3] = -l2[i]
            l2A[4*i+2, 4*i] = -l2[i]
            l2A[4*i+3, 4*i+1] = l2[i]
        

        U = linalg.expm(-l2A) @ R @ linalg.expm(-l1A)
        np.set_printoptions(linewidth=140, precision=3)
                
        T, Z = linalg.schur(U, output="real", overwrite_a=False)
        
        exponent = np.zeros_like(majs[0])

        for i in range(qubits):
            if T[2*i, 2*i+1] < 0:
                T[2*i, 2*i+1] *= -1
                T[2*i+1, 2*i] *= -1
                Z[:,[2*i, 2*i+1]]  = Z[:,[2*i+1,2*i]]
                
            theta = np.arctan2(T[2*i+1, 2*i], T[2*i, 2*i])
            exponent += (1/2.)*theta*majs[2*i] @ majs[2*i+1]
        #print("python schur", file=sys.stderr)
        #print(Z, file=sys.stderr)
        #print("python phase: ", linalg.expm(exponent)[0,0], file=sys.stderr)
        
        obj = {"type": "aka_kak",
               "pythonValR": linalg.expm(exponent)[0,0].real,
               "pythonValI": linalg.expm(exponent)[0,0].imag,
               "qubits": qubits,
               "R": list(R.T.reshape(2*qubits*2*qubits)),
               "lambda1": l1,
               "lambda2": l2,
               "phaseReal": phase.real,
               "phaseImag": phase.imag,
               }
        print(json.dumps(obj))


        
def make_passive_decomp_tests(seed=1000, count=2, qubits = 4):
    rng = default_rng(seed)
    
    #print(count)
    
    for _ in range(count):        
        A = rng.random((qubits,qubits), dtype=np.float64)
        B = rng.random((qubits,qubits), dtype=np.float64)
        A = (A - A.T)/2
        B = (B + B.T)/2
        alpha = (np.kron(A, np.eye(2,dtype=np.float64)) + np.kron(B, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R = linalg.expm(-alpha)
        phase = np.exp(-(1.j/4.)*(alpha @ np.kron(np.eye(qubits,dtype=complex), np.array([[0,1],[-1,0]],dtype=complex))).trace())       
        majs = compute_majs(qubits)

        exponent = np.zeros_like(majs[0])
        for i in range(2*qubits):
            for j in range(2*qubits):
                exponent += (alpha[i][j]/4.) * majs[i]@majs[j]

        obj = {"type": "passive-decomp", 
               "qubits": qubits,
               "R": list(R.T.reshape(2*qubits*2*qubits)),
               "phaseReal": phase.real,
               "phaseImag": phase.imag,
               }
        print(json.dumps(obj))



def make_comp_basis_inner_product_tests(seed=1000, count=2, paired_qubits=False, qubits = 4):
    rng = default_rng(seed)
    
    #print(count)
    
    for _ in range(count):        
        A = rng.random((qubits,qubits), dtype=np.float64)
        B = rng.random((qubits,qubits), dtype=np.float64)
        A = (A - A.T)/2
        B = (B + B.T)/2
        alpha = (np.kron(A, np.eye(2,dtype=np.float64)) + np.kron(B, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R = linalg.expm(-alpha)
        phase = np.exp(-(1.j/4.)*(alpha @ np.kron(np.eye(qubits,dtype=complex), np.array([[0,1],[-1,0]],dtype=complex))).trace())

        l = rng.random(qubits//2, dtype=np.float64)

        c_vec = np.zeros(2*qubits, dtype=int)
        short_c_vec = None
        if paired_qubits == True:
            short_c_vec = rng.integers(0,2,qubits//2)
            short_c_vec = np.kron(short_c_vec, np.array([1,1],dtype=int))
        else:
            short_c_vec = rng.integers(0,2,qubits)
            if sum(short_c_vec) % 2 != 0:
                short_c_vec[0] ^= 1
        for i in range(qubits):
            c_vec[2*i] = short_c_vec[i]


        majs = compute_majs(qubits)
        exponent = np.zeros((2**qubits, 2**qubits), dtype=complex)            
        for i in range(2*qubits):
            for j in range(2*qubits):
                exponent += majs[i]@majs[j]*alpha[i][j]/4
        K = linalg.expm(exponent)

        vec = np.zeros(2**qubits,dtype=complex)
        vec[0] = 1
        for i, li in enumerate(l):
            vec = linalg.expm(li*(majs[4*i]@majs[4*i+2] -majs[4*i+1]@majs[4*i+3])/2.) @ vec        
        vec = K @ vec

        y = 0
        for i, c in enumerate(c_vec):
            if c != 0:
                y ^= (1<<( (qubits-1-(i// 2)) ))

        val = vec[y]
        obj = {"type": "cb-inner-product", 
               "qubits": qubits,
               "R": list(R.T.reshape(2*qubits*2*qubits)),
               "l": list(l),
               "c_vec": list(map(int,c_vec)),
               "phaseReal": phase.real,
               "phaseImag": phase.imag,
               "pythonvalReal": val.real,
               "pythonvalImag": val.imag
               }
        print(json.dumps(obj))
        

def make_two_flo_state_inner_prod_tests(seed=1000, count=2, qubits = 4):
    rng = default_rng(seed)
    import json
    #print(count)
    majs = compute_majs(qubits)
    for _ in range(count):        
        A1 = rng.random((qubits,qubits), dtype=np.float64)
        B1 = rng.random((qubits,qubits), dtype=np.float64)
        A1 = (A1 - A1.T)/2
        B1 = (B1 + B1.T)/2
        alpha1 = (np.kron(A1, np.eye(2,dtype=np.float64)) + np.kron(B1, np.array([[0,1],[-1,0]],dtype=np.float64)))

        A2 = rng.random((qubits,qubits), dtype=np.float64)
        B2 = rng.random((qubits,qubits), dtype=np.float64)
        A2 = (A2 - A2.T)/2
        B2 = (B2 + B2.T)/2
        alpha2 = (np.kron(A2, np.eye(2,dtype=np.float64)) + np.kron(B2, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R1 = linalg.expm(-alpha1)
        R2 = linalg.expm(-alpha2)
        
        l1 = rng.random(qubits//2, dtype=np.float64)
        l2 = rng.random(qubits//2, dtype=np.float64)
        #a = np.zeros((2*qubits, 2*qubits), dtype=float)
        #for i in range(qubits//2):
        #    a[4*i][4*i+2] = l1[i]
        #    a[4*i+2][4*i] = -l1[i]
        #    a[4*i+1][4*i+3] = -l1[i]
        #    a[4*i+3][4*i+1] = l1[i]


            
        exponent1 = np.zeros((2**qubits, 2**qubits), dtype=complex)
        exponent2 = np.zeros((2**qubits, 2**qubits), dtype=complex)
        for i in range(2*qubits):
            for j in range(2*qubits):
                exponent1 += majs[i]@majs[j]*alpha1[i][j]/4.
                exponent2 += majs[i]@majs[j]*alpha2[i][j]/4.
                
        K1 = linalg.expm(exponent1)
        K2 = linalg.expm(exponent2) 

        phase1 = K1[0][0]
        phase2 = K2[0][0]

        vec = np.zeros(2**qubits,dtype=complex)
        vec[0] = 1
        for i, li in enumerate(l2):
            vec = linalg.expm(li*(majs[4*i]@majs[4*i+2] -majs[4*i+1]@majs[4*i+3])/2.) @ vec
        
        vec = K1.conjugate().T @  K2 @ vec
        for i, li in enumerate(l1):
            vec = linalg.expm(-li*(majs[4*i]@majs[4*i+2] -majs[4*i+1]@majs[4*i+3])/2.) @ vec
        correct_prod = vec[0]
        obj = {"type": "flo-inner-product",
               "qubits": qubits,
               "phase1R":K1[0][0].real,
               "phase1I":K1[0][0].imag,
               "phase2R":K2[0][0].real,
               "phase2I":K2[0][0].imag,
               "R1": list(R1.T.reshape(2*qubits*2*qubits)),
               "R2": list(R2.T.reshape(2*qubits*2*qubits)),
               "prodR": correct_prod.real,
               "prodI": correct_prod.imag,
               "l1": list(l1),
               "l2": list(l2)
               }
        print(json.dumps(obj),end="")

def make_MKA_test(seed=1000, count=2, qubits = 4):
    majs = compute_majs(qubits)

    rng = default_rng(seed)
    import json
    #print(count)
    majs = compute_majs(qubits)
    for _ in range(count):        
        A = rng.random((qubits,qubits), dtype=np.float64)
        B = rng.random((qubits,qubits), dtype=np.float64)
        A = (A - A.T)/2
        B = (B + B.T)/2
        alpha = (np.kron(A, np.eye(2,dtype=np.float64)) + np.kron(B, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R = linalg.expm(-alpha)

        exponent = np.zeros_like(majs[0])
        for i in range(2*qubits):
            for j in range(2*qubits):
                exponent += (alpha[i][j]/4.)*majs[i]@majs[j]
        passive = linalg.expm(exponent)

        # now generate a random FLO unitary so we can construct M
        A = rng.random((2*qubits, 2*qubits), dtype=np.float64)
        A = (A - A.T)/2.
        RA = linalg.expm(-A)
        exponent = np.zeros_like(majs[0])
        for i in range(2*qubits):
            for j in range(2*qubits):
                exponent += (A[i][j]/4.)*majs[i]@majs[j]
        U = linalg.expm(exponent)
            
        # U|0><0|U^\dagger K A = <0| U^\dagger KA U |0>
        
        active_lambdas = rng.random(qubits//2, dtype=np.float64)

        exponent = np.zeros_like(majs[0])
        for i in range(qubits//2):
            exponent += (active_lambdas[i]/2.)*(majs[4*i]@majs[4*i+2] - majs[4*i+1]@majs[4*i+3])
        activeU = linalg.expm(exponent)

        correct_prod = (U.conjugate().transpose() @ passive @ activeU @ U)[0,0]

        M = np.zeros((2*qubits, 2*qubits), dtype=np.float64)
        for i in range(qubits):
            M[2*i][2*i+1] = 1
            M[2*i+1][2*i] = -1
        M = RA.T @ M @ RA
        obj = {"type": "mka_prod",
               "qubits": qubits,
               "M": list(M.T.reshape(2*qubits*2*qubits)),
               "R": list(R.T.reshape(2*qubits*2*qubits)),
               "RphaseR": passive[0,0].real,
               "RphaseI": passive[0,0].imag,
               "prodR": correct_prod.real,
               "prodI": correct_prod.imag,
               "A": list(active_lambdas),
               }
        print(json.dumps(obj),end="")
        
if __name__ == "__main__":

    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inner-product',default="0",type=int, help="number of FLO inner products to test (defaults to 0)")
    parser.add_argument('-c', '--cb-inner-product',default="0",type=int, help="number of computational basis inner products to test (defaults to 0)")
    parser.add_argument('-d', '--decompose-passive',default="0",type=int, help="number passive FLO unitary decompositions to test (defaults to 0)")
    parser.add_argument('-o', '--symplectic-orthogonal',default="0",type=int, help="number symplectic orthogonal factorizations to test (defaults to 0)")
    parser.add_argument('-k', '--aka-kak',default="0",type=int, help="number of aka to kak calculations to test (defaults to 0)")
    parser.add_argument('-m', '--mka',default="0",type=int, help="number of aka to mka products to test (defaults to 0)")
    parser.add_argument('-s', '--seed',default="1000",type=int, help="random seed")
    parser.add_argument('-q', '--qubits',default="4",type=int, help="number of qubits")
    args = parser.parse_args(sys.argv[1:])

    if args.inner_product > 0:
        make_two_flo_state_inner_prod_tests(seed=args.seed, count = args.inner_product, qubits=args.qubits)
    if args.cb_inner_product > 0:
        make_comp_basis_inner_product_tests(seed=args.seed, count = args.cb_inner_product, paired_qubits=True, qubits=args.qubits)
    if args.decompose_passive > 0:
        make_passive_decomp_tests(seed=args.seed, count=args.decompose_passive, qubits=args.qubits)
    if args.symplectic_orthogonal > 0:
        make_symplectic_orthogonal_decompostion_test(seed=args.seed, count=args.symplectic_orthogonal, qubits=args.qubits)
    if args.aka_kak > 0:
        make_aka_kak_test(seed=args.seed, count=args.aka_kak, qubits=args.qubits)
    if args.mka > 0:
        make_MKA_test(seed=args.seed, count=args.mka, qubits=args.qubits)
        

#! /usr/bin/env python3
import numpy as np
from pfapack import pfaffian as pf
from numpy.random import default_rng
from scipy import linalg
import sys 
import json

qubits = 8

def compute_majs():
    pI = np.eye(2, dtype=complex)
    pX = np.array([[0,1],[1,0]], dtype=complex)
    pY = np.array([[0,-1j],[1j,0]], dtype=complex)
    pZ = np.array([[1,0],[0,-1]], dtype=complex)
    
    majs = []
    
    
    for i in range(qubits):
        even = pX
        odd = pY
        
        for j in range(i):
            even = np.kron(even,pZ)
            odd = np.kron(odd,pZ)
        for j in range(i+1, qubits):
            even = np.kron(pI,even)
            odd = np.kron(pI,odd)
        majs.append(even)
        majs.append(odd)
    return majs


def make_passive_decomp_tests(seed=1000, count=2):
    rng = default_rng(seed)
    
    #print(count)
    
    for _ in range(count):        
        A = rng.random((qubits,qubits), dtype=np.float64)
        B = rng.random((qubits,qubits), dtype=np.float64)
        A = (A - A.T)/2
        B = (B + B.T)/2
        alpha = (np.kron(A, np.eye(2,dtype=np.float64)) + np.kron(B, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R = linalg.expm(alpha)
        phase = np.exp(-(1.j/4.)*(alpha @ np.kron(np.eye(qubits,dtype=complex), np.array([[0,1],[-1,0]],dtype=complex))).trace())       

        
        obj = {"type": "passive-decomp", 
               "qubits": qubits,
               "R": list(R.T.reshape(2*qubits*2*qubits)),
               "phaseReal": phase.real,
               "phaseImag": phase.imag,
               }
        print(json.dumps(obj))



def make_comp_basis_inner_product_tests(seed=1000, count=2, paired_qubits=False):
    rng = default_rng(seed)
    
    #print(count)
    
    for _ in range(count):        
        A = rng.random((qubits,qubits), dtype=np.float64)
        B = rng.random((qubits,qubits), dtype=np.float64)
        A = (A - A.T)/2
        B = (B + B.T)/2
        alpha = (np.kron(A, np.eye(2,dtype=np.float64)) + np.kron(B, np.array([[0,1],[-1,0]],dtype=np.float64)))

        R = linalg.expm(alpha)
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

        y = 0
        for i, c in enumerate(c_vec):
            if c != 0:
                y ^= (1<<(i//2))
                       
        majs = compute_majs()
        exponent = np.zeros((2**qubits, 2**qubits), dtype=complex)            
        for i in range(2*qubits):
            for j in range(i):
                exponent += majs[i]@majs[j]*alpha[i][j]/2                
        K = linalg.expm(exponent)
        
        vec = np.zeros(2**qubits,dtype=complex)
        vec[0] = 1
        for i, li in enumerate(l):
            vec = linalg.expm(li*(majs[4*i]@majs[4*i+2] -majs[4*i+1]@majs[4*i+3])/2.) @ vec        
        vec = K @ vec
        
        y = 0
        for i, c in enumerate(c_vec):
            if c != 0:
                y ^= (1<<(i//2))
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
        

def make_two_flo_state_inner_prod_tests(seed=1000, count=2):
    rng = default_rng(seed)
    import json
    #print(count)
    majs = compute_majs()
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

        R1 = linalg.expm(alpha1)
        R2 = linalg.expm(alpha2)
        
        l1 = rng.random(qubits//2, dtype=np.float64)
        l2 = rng.random(qubits//2, dtype=np.float64)
        a = np.zeros((2*qubits, 2*qubits), dtype=float)
        for i in range(qubits//2):
            a[4*i][4*i+2] = l1[i]
            a[4*i+2][4*i] = -l1[i]
            a[4*i+1][4*i+3] = -l1[i]
            a[4*i+3][4*i+1] = l1[i]


            
        exponent1 = np.zeros((2**qubits, 2**qubits), dtype=complex)
        exponent2 = np.zeros((2**qubits, 2**qubits), dtype=complex)
        for i in range(2*qubits):
            for j in range(2*qubits):
                exponent1 += majs[i]@majs[j]*alpha1[i][j]/4
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inner-product',default="0",type=int, help="number of FLO inner products to test (defaults to 0)")
    parser.add_argument('-c', '--cb-inner-product',default="0",type=int, help="number of computational basis inner products to test (defaults to 0)")
    parser.add_argument('-d', '--decompose-passive',default="0",type=int, help="number passive FLO unitary decompositions to test (defaults to 0)")
    parser.add_argument('-s', '--seed',default="1000",type=int, help="random seed")
    
    args = parser.parse_args(sys.argv[1:])

    if args.inner_product > 0:
        make_two_flo_state_inner_prod_tests(seed=args.seed, count = args.inner_product)
    if args.cb_inner_product > 0:
        make_comp_basis_inner_product_tests(seed=args.seed, count = args.cb_inner_product, paired_qubits=True)
    if args.decompose_passive > 0:
        make_passive_decomp_tests(seed=args.seed, count=args.decompose_passive)

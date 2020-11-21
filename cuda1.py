from numba import vectorize, jit, cuda 
import numpy as np 
# to measure exec time
from timeit import default_timer as timer 

# function optimized to run on CPU 
@vectorize(['float64(float64)'], target ="cpu")                         
def func1(x): 
    return x+1

# function optimized to run on GPU 
@vectorize(['float64(float64)'], target ="cuda")                         
def func2(x): 
    return x+1

# kernel to run on GPU
@cuda.jit
def func3(a, N):
    tid = cuda.grid(1)
    if tid < N:
        a[tid] += 1


if __name__=="__main__": 
    n = 100000000
    a = np.ones(n, dtype = np.float64)

    times = []
    for i in range(0,5):
         start = timer() 
         a=func1(a) 
         finish = timer()-start
         times.append(finish)
         print(i, " Vectorized for CPU ufunc : t=", round(finish,6), "  --> first 5 results: ", a[0:5])  
    print("Average time:             ", sum(times)/len(times) )
    print("Average after first pass: ", sum(times[1:4])/len(times[1:4]), "\n" )

    times = []
    for i in range(0,5):
         start = timer() 
         a=func2(a) 
         finish = timer()-start
         times.append(finish)
         print(i, "Vectorized for GPU ufunc: t=", round(finish,6), "  --> first 5 results: ", a[0:5])  
    print("Average time:             ", sum(times)/len(times) )
    print("Average after first pass: ", sum(times[1:4])/len(times[1:4]), "\n" )

    times = []
    threadsperblock = 1024
    blockspergrid = (a.size + (threadsperblock - 1)) // threadsperblock
    for i in range(0,5):
         start = timer() 
         func3[blockspergrid, threadsperblock](a, n) 
         finish = timer()-start
         times.append(finish) 
         print(i, "Using GPU kernel: t=", round(finish,6), "  --> first 5 results: ", a[0:5])  
    print("Average time:             ", sum(times)/len(times) )
    print("Average after first pass: ", sum(times[1:4])/len(times[1:4]), "\n" )
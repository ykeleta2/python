from numba import cuda, numba
import numpy as np,csv,pandas as pd
from distances import Distances

@cuda.jit
def get_string_distances(a,c,):
	row, col = cuda.grid(2)
	if row < a.shape[0] and col < a.shape[1]:
		c[row,col] = string_distance(a[row],a[col])
    	    

@cuda.jit('int64(float64[:],float64[:])',device=True) 
def string_distance(s1,s2):
	m=len(s1) 
	n=len(s2) 

	v1=cuda.local.array(shape=1,dtype=numba.float64)
	v2=cuda.local.array(shape=1,dtype=numba.float64)

	for i in range(0,n):
		v1[i]=i

    # Initialize the table
    # Populate the table using dynamic programming
	for i in range(0, m-1):
		v2[0]=i+1
		for j in range(0, n-1):
			deleteCost=v1[j+1] + 1
			insertCost=v2[j] + 1
			 
			if s1[i] == s2[j]:
				substitutionCost=v1[j]
			else:
				substitutionCost=v1[j]+1

			v2[j+1]=min(deleteCost,insertCost,substitutionCost)
        
	v1=v2
	return v1[n-1]

def find_distance():
	threads_per_block = 256
	blocks_per_grid = 256#(N + threads_per_block - 1) // threads_per_block

	# Copy data to the device
	a=[1,2,2,3,6]
	b=[4,5,7,1,2]
	c=[4,5,7,3,1]
	d=[1,3,2,2,5]
 
	
	#a1=np.array([a,b,c,d])
	#print("type is: ",type(a1))
	dist = Distances()
	a1=dist.get_spikes_as_num_array()
	print("type is: ",type(a1))
	print("shape: (",a1.shape[0],a1.shape[1],")")
	#a1=np.ndarray(len([a,b,c,d]), dtype=np.ndarray)
	#a1=cuda.to_device(a)
	#a2=cuda.to_device(b)
	#gpu_results=cuda.to_device(np.ndarray(shape=(a1.shape[0],a1.shape[0])))
	
	gpu_results=cuda.to_device(np.ndarray(shape=(len(a1),len(a1[0]))))
	# Launch the kernel
	get_string_distances[blocks_per_grid, threads_per_block](a1,gpu_results)
	res = gpu_results.copy_to_host()
	print("Got ",len(res), " results")
	print("shape: ",res.shape)
 

find_distance()
 

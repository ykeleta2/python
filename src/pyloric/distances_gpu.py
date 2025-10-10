from numba import cuda, numba
import numpy as np,csv,pandas as pd
from distances import Distances


#class GPUDistance:
	#def __init__(self):
	#	self.dir ="src/pyloric/data"
	#	self.fname = "converted_temp.csv"  


@cuda.jit
def get_string_distances(a,c,):
	row, col = cuda.grid(2)
	if row < a.shape[0] and col < a.shape[1]:
		c[row,col] = string_distance(a[row],a[col])
			

@cuda.jit('int64(int64[:],int64[:])',device=True) 
def string_distance(s1,s2):
	m=len(s1) 
	n=len(s2) 

	v1=cuda.local.array(shape=1,dtype=numba.int64)
	v2=cuda.local.array(shape=1,dtype=numba.int64)

	for i in range(0,n):
		v1[i]=i

	for j in range(0,m):
		v2[j]=0

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

def convert_to_num_array(spike_pattern):
	converted_patterns=list()
	for i,pattern in spike_pattern.items():
		#num_array=(self.convert_to_symbol(pattern))
		converted_patterns.append(num_array)
		
	return np.array(converted_patterns)

def read_converted_spikes():
	full_name = "src/pyloric/data/converted_spike_patterns.csv" 

	df=pd.read_csv(full_name,dtype={'converted_spike_pattern': 'string'},skiprows=0)
	df = df.reset_index()

	lst=list()
	for i, row in df.iterrows():
		pattern = df.at[i,'converted_spike_pattern']
		#print(convert_to_array(pattern))
		lst.append(convert_to_array(pattern))
		#length = 0 if pd.isnull(pattern) else len(pattern)

	#return df.iloc[:,4].values #, max_length   
	return lst	

def convert_to_array(pattern):
	lst=list()
	for s in pattern:
		lst.append(int(s))
	
	return np.array(lst)


def run_calculation():

	#a=[1,2,2,3,6,6]
	#b=[4,5,7,1,2,6]
	c=[4,5,7,3,1,7]
	d=[1,3,2,2,5,7]
	e=[1,3,2,2,6,7]	
	a=[1,1]
	b=[1,1]
	a1=np.array([a,b])
 
	# Write out output file shell
	with open('src/pyloric/data/output.csv', mode='w', newline='') as empty_file:
		pass

	input_file='src/pyloric/data/converted_temp.csv'
	
	#for chunk in pd.read_csv(input_file,chunksize=10): #i, df in enumerate(pd.read_csv(input_file, chunksize=10)): 
	#	lst=list() 
	#		for converted_spike_pattern in chunk['converted_spike_pattern']:			 
	#			lst.append(convert_to_array(converted_spike_pattern))

	#		a = np.array(lst)	
	send_chunk_to_cpu(a1)
		#print(lst)
	#a1=np.array(read_converted_spikes())
	#print("type is: ",type(a1))
	#print("shape: (",a1.shape[0],a1.shape[1],")")
	#a1=np.ndarray(len([a,b,c,d]), dtype=np.ndarray)
	#a1=cuda.to_device(a)
	#a2=cuda.to_device(b)
	#gpu_results=cuda.to_device(np.ndarray(shape=(a1.shape[0],a1.shape[0])))
	
def send_chunk_to_cpu(a1):
	threads_per_block = 256
	blocks_per_grid = 256#(N + threads_per_block - 1) // threads_per_block
	
	gpu_results=cuda.to_device(np.ndarray(shape=(len(a1),len(a1)),dtype=int))
	
	# Launch the kernel
	get_string_distances[blocks_per_grid, threads_per_block](a1,gpu_results)
	res = gpu_results.copy_to_host()
	print(type(res))
	print("Got ",len(res), " results")
	print("shape: ",res.shape)
	print(res)
	#with open('src/pyloric/data/output.csv', mode='a', newline='') as output_file:
	#	writer=csv.writer(output_file)
	#	for x in res:
	#		writer.writerow(x)
	

#find_distance()
run_calculation()
#print(read_converted_spikes())

from numba import cuda, numba
import numpy as np,csv,pandas as pd
import time

#class GPUDistance:
	#def __init__(self):
	#	self.dir ="src/pyloric/data"
	#	self.fname = "converted_temp.csv"  


@cuda.jit
def get_string_distances(a,x,y,output,):
	row, col = cuda.grid(2)
	l=len(a)
	if row < x and col < y:
		s1_coord=int(row/l+row%l)
		s2_coord=int(col/l+col%l)
		output[row,col] = string_distance(a[s1_coord],a[s2_coord]) # c[row][col] ??
			

@cuda.jit('int64(int64[:],int64[:])',device=True) 
def string_distance(s1,s2):
	m=len(s1)+1 
	n=len(s2)+1 
	#print("m: ",m," n: ",n)

	#v1=cuda.local.array(shape=1,dtype=numba.int64)
	#v2=cuda.local.array(shape=1,dtype=numba.int64)

	d=cuda.local.array(shape=(91,91),dtype=numba.int64)
	#d=np.zeros(shape=(91,91),dtype=int)
	#for i in range(0,n):
	#	v1[i]=i

	#for j in range(0,m):
	#	v2[j]=0
	#print("d: ",d)
	for i in range(1,m):
		d[i,0] = i

	for j in range(1,n):
		d[0,j] = j

	# Initialize the table
	# Populate the table using dynamic programming
	for j in range(1, n):
		#v2[0]=i+1
		for i in range(1, m):
			#deleteCost=v1[j+1] + 1
			#insertCost=v2[j] + 1
			#print(s1[i-1]," == ",s2[j-1]," ?")
			if s1[i-1] == s2[j-1]:
				substitutionCost=0
			else:
				substitutionCost=1

			d[i,j]=min(d[i-1, j] + 1,d[i, j-1] + 1,d[i-1, j-1] + substitutionCost)
		
	#v1=v2
	#print(d)
	return d[m-1,n-1]

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
	#lst=list()
	#for s in pattern:
	#	lst.append(int(s))
	
	#return np.array(lst)
	return [int(x) for x in str(pattern)]

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

	input_file='src/pyloric/data/converted_spike_patterns.csv'
	
	chunk_size=5000
	for chunk in pd.read_csv(input_file,chunksize=chunk_size): #i, df in enumerate(pd.read_csv(input_file, chunksize=10)): 
		lst=list() 
		print("One time!")
		for converted_spike_pattern in chunk['converted_spike_pattern']:	
		#	print(" -> ",converted_spike_pattern)		 
			lst.append(convert_to_array(converted_spike_pattern))

		a1 = np.array(lst)	
	#print(a1)
	t0 = time.time()
	print(len(a1))
	#send_chunk_to_gpu(a1)
	t1 = time.time()
	print(t1-t0)
		#print(lst)
	#a1=np.array(read_converted_spikes())
	#print("type is: ",type(a1))
	#print("shape: (",a1.shape[0],a1.shape[1],")")
	#a1=np.ndarray(len([a,b,c,d]), dtype=np.ndarray)
	#a1=cuda.to_device(a)
	#a2=cuda.to_device(b)
	#gpu_results=cuda.to_device(np.ndarray(shape=(a1.shape[0],a1.shape[0])))
	
def send_chunk_to_gpu(a1):
	threads_per_block = (16,16)
	blocks_per_grid = 16#(N + threads_per_block - 1) // threads_per_block
	print("Processing a vector of length: ",len(a1))
	gpu_results=cuda.to_device(np.zeros(shape=(len(a1)**2,len(a1)**2),dtype=int))
	
	# Launch the kernel
	get_string_distances[blocks_per_grid, threads_per_block](a1,len(a1)**2,len(a1)**2,gpu_results)
	res = gpu_results.copy_to_host()
	print(type(res))
	print("Got ",len(res), " results")
	print("shape: ",res.shape)
	print(res)
	with open('src/pyloric/data/output.csv', mode='a', newline='') as output_file:
		writer=csv.writer(output_file)
		for x in res:
			writer.writerow(x)
	
def test_gpu_standalone():
	#with open('src/pyloric/data/output.csv', mode='w', newline='') as empty_file:
	#	pass

	input_file='src/pyloric/data/converted_temp3.csv'
	
	for chunk in pd.read_csv(input_file,chunksize=2): #i, df in enumerate(pd.read_csv(input_file, chunksize=10)): 
		lst=list() 
		for converted_spike_pattern in chunk['converted_spike_pattern']:	
			print(" -> ",converted_spike_pattern)		 
			lst.append(convert_to_array(converted_spike_pattern))

		a1 = np.array(lst)	
	print(a1)
	string_distance(a1[1],a1[1])

#find_distance()
run_calculation()
#test_gpu_standalone()


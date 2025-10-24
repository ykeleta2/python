from numba import cuda, numba
import numpy as np,csv,pandas as pd
import time

#class GPUDistance:
	#def __init__(self):
	#	self.dir ="src/pyloric/data"
	#	self.fname = "converted_temp.csv"  

grid=300

@cuda.jit
def get_string_distances(a,output):
	row, col = cuda.grid(2)
	l=len(a)
	if row < l and col < l and col <= row:
		output[row,col] = string_distance(a[row],a[col])  
			

@cuda.jit('int32(int16[:],int16[:])',device=True) 
def string_distance(s1,s2):
	m=len(s1)+1 
	n=len(s2)+1 
 
	# Unfortunately, we can't use a parameter for the size of the table.
	d=cuda.local.array(shape=(295,295),dtype=numba.int32)

	for i in range(1,m):
		d[i,0] = i

	for j in range(1,n):
		d[0,j] = j

	# Initialize the table
	# Populate the table using dynamic programming
	for j in range(1, n):
		for i in range(1, m):
			if s1[i-1] == s2[j-1]:
				substitutionCost=0
			else:
				substitutionCost=1

			d[i,j]=min(d[i-1, j] + 1,d[i, j-1] + 1,d[i-1, j-1] + substitutionCost)
		
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
		lst.append(convert_to_array(pattern))


	#return df.iloc[:,4].values #, max_length   
	return lst	

def convert_to_array(pattern):
	return [int(x) for x in str(pattern)]

 

def run_calculation():

	# Write out output file shell
	with open(f'src/pyloric/data/distance_matrix_{grid}x{grid}.csv', mode='w', newline='') as empty_file:
		pass


	input_file=f'src/pyloric/data/converted_spike_patterns_{grid}x{grid}.csv'  
	
	chunk_size=10000
	lst=list() 
	for chunk in pd.read_csv(input_file,header=0,chunksize=chunk_size):  #i, df in enumerate(pd.read_csv(input_file, chunksize=10)): 
		for i,row in chunk.iterrows(): # converted_spike_pattern in chunk['converted_spike_pattern']:	
			#	print(" -> ",converted_spike_pattern)	
			lst.append(convert_to_array(row.iloc[2])) #(row[2]))
		print("List length: ",len(lst))
	
	a1 = np.array(lst,dtype=np.int16)	
	#print(a1)
	t0 = time.time()
	print(len(a1))
	send_chunk_to_gpu(a1)
	t1 = time.time()
	print("Total time on GPU: ",t1-t0)
	
def send_chunk_to_gpu(a1):
	threads_per_block = (32,32)
	blocks_per_grid = (2048,2048)#(N + threads_per_block - 1) // threads_per_block
	print("Processing a vector of length: ",len(a1))
	gpu_results=cuda.to_device(np.zeros(shape=(len(a1),len(a1)),dtype=np.int16))
	
	# Launch the kernel
	get_string_distances[blocks_per_grid, threads_per_block](a1,gpu_results)
	res = gpu_results.copy_to_host()
	print(type(res))
	print("Got ",len(res), " results")
	print("shape: ",res.shape)
	print(res)
	with open(f'src/pyloric/data/distance_matrix_{grid}x{grid}.csv', mode='a', newline='') as output_file:
		writer=csv.writer(output_file)
		for x in res:
			writer.writerow(x)

def convert_to_symmetrical_matrix():
	df=pd.read_csv(f'src/pyloric/data/distance_matrix_{grid}x{grid}.csv',header=None)
	mat=df.to_numpy()
	res=mat + mat.T
	print("writing matrix ",res.shape)
	with open(f'src/pyloric/data/output_{grid}x{grid}_symmetrical.csv',mode='w',newline='') as out_file:
		writer=csv.writer(out_file)
		writer.writerows(res)

def run_test():

	#a=[1,2,2,3,6,6]
	#b=[4,5,7,1,2,6]
	c=[4,5,7,3,1,7]
	d=[1,3,2,2,5,7]
	e=[1,4,1,1,2,4,2,2,1,1,1,2,4,2,2,1,2,1,1,4,2,4,2,2,1,1,1,2,4,2,2,1,1,1,2,4,3,2,1,1,0]	 #41 chars
	f=[1,4,1,1,2,4,2,2,1,1,1,2,4,2,2,1,2,1,1,4,2,4,2,2,1,1,1,2,4,2,2,1,1,1,2,4,0,0,0,0,0]

	g="10000000000000000000000000000000000000000000000000003111111232323232323232131111112323232323232321311111123232323232323213111111232323232323232131111112323232323232321311111123232323232000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
	h="10000000000000000000000000000000000000000000000000111111133333333333333311111112222222222222222222221111111133333333333333311111112222222222222222222221111111133333333333333311111112222222222222222200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
	#print("g length: ",len(convert_to_array(g))," h length: ",len(convert_to_array(h)))
	a1=np.array([convert_to_array(g),convert_to_array(h)],dtype=np.int16)
	send_chunk_to_gpu(a1)
	
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
#run_calculation()
run_test()
#convert_to_symmetrical_matrix()
#test_gpu_standalone()


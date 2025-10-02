import csv,pandas as pd, numpy as np


class Distances:
      
    def __init__(self):
        self.dir="src/pyloric/data"
        self.fname="combined_spike_patterns.csv" #"temp_patterns.csv" 
    
    def read_file(self):
        full_name=self.dir+'/'+self.fname
        #input = np.loadtxt(full_name, dtype='i', delimiter=',')
        #with open(full_name) as csvfile:
         #   reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         #   for row in reader:
         #       print(row[0])
               # print(', '.join(row[0]))
        df=pd.read_csv(full_name,dtype={'spike_patterns': 'string'},skiprows=0)
        df = df.reset_index()

        # Update empty rows (no spikes) to some default value
        # Get max length while you're at it
        max_length=0
        for i, row in df.iterrows():
            pattern = df.at[i,'spike_patterns']
            length = 0 if pd.isnull(pattern) else len(pattern)
            
            if length > max_length:
                max_length = length

            if  pd.isna(df.at[i,'spike_patterns']):
                df.at[i,'spike_patterns'] = 'x' 
 
        return df.iloc[:,3].values, max_length

     #def calculate_pairwise(self,v):
     #   n=len(v)
     #   print(n*n)
     #   results=np.zeros((n,n))
       # print(n)
     #   for i in range(0,n):
     #       for j in range(0,i):
     #           if (i*j)%1000==0:
     #               print(i,",",j,i*j)

     #           results[i,j]=damerau_levenshtein_distance(v[i],v[j])

     #   print("done")
     #   return results
        """
        function calculate_pairwise(vec,l=false)

            n=length(vec)
            results=zeros(Int32,n,n)
            Threads.@threads for i in 1:n 

                @inbounds for j in 1:i 
                    results[j,i]=DamerauLevenshtein()(vec[i],vec[j])
                end
            end
            Symmetric(results)
        end
        """
    def get_spikes_as_num_array(self):
        spike_patterns, max_length = self.read_file()
       
        print("length: ",max_length)
        #patterns = np.ndarray(len(spike_patterns), dtype=np.ndarray)
        #patterns=np.array(len(spike_patterns),dtype=int)
        patterns=list()
        for i in range(0,len(spike_patterns)):
            symbol = spike_patterns[i]
            
            num_array=(self.convert_to_symbol(symbol))
            #print(type(num_array))
            if len(num_array) > max_length:
                print("offending one: ", i)

            num_array=np.pad(num_array,(0,max_length-len(num_array)),'constant')
            #print(num_array)
            #patterns[i]=num_array
            patterns.append(num_array)
        return np.array(patterns)
    
    def convert_to_symbol(self,symbol):
        num_array=np.zeros(len(symbol))

        for i in range(0,len(symbol)):
            if symbol[i] == "a":
                num_array[i] = 1
            elif symbol[i] == 'l':
                num_array[i] = 2
            elif symbol[i] == 'p':
                num_array[i] = 3
            elif symbol[i] == 't':
                num_array[i] = 4
            elif symbol[i] == 'x':
                num_array[i] = -1
            else:
                num_array[i] = -100
       # print(num_array)
        return num_array
        
    def run(self):
        vec=self.read_file()
        #print(vec)
    #    res=self.calculate_pairwise(vec)
       # print(res)


#d=damerau_levenshtein_distance('smtih', 'smiadfadsfasdfth')  # expected result: 1
#print(d)
dist=Distances()
#print(vec)
print(dist.get_spikes_as_num_array())
#print(type(dist.get_spikes_as_num_array()[1]))
#arr=dist.get_spikes_as_num_array()[1]
#print(np.pad(arr,(0,90-len(arr)),'constant'))
#dist.get_spikes_as_num_array()
 
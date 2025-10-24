import csv,pandas as pd, numpy as np


class Distances:
      
    def __init__(self):
        self.dir ="src/pyloric/data"
        self.grid = "100"
        self.fname =  f"combined_spike_patterns_{self.grid}x{self.grid}_3.csv"  #"temp_patterns2.csv" 

    
    def read_file(self):
        full_name=self.dir+'/'+self.fname
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
    
    def handle_pattern(self):
        output_fname="converted_spike_patterns"
        with open(self.dir+'/'+f'{output_fname}_{self.grid}x{self.grid}_3.csv', mode='w', newline='', encoding="utf-8") as empty_file:
            writer = csv.writer(empty_file)
            writer.writerow(["α_fast","β_fast","converted_spike_pattern"])
        
        input_fname=self.dir+'/'+self.fname

        df=pd.read_csv(input_fname)
        max_length=df.spike_patterns.dropna().map(len).max()  
        print("max length: ",max_length)

        
        for i, df in enumerate(pd.read_csv(input_fname, chunksize=10000)):  
 
            df['converted_spike_pattern']=df['spike_patterns'].str.replace('a','1').str.replace('l','2').str.replace('p','3').str.replace('t','0')
            df['converted_spike_pattern']=df['converted_spike_pattern'].fillna('0')
            df['converted_spike_pattern']=df['converted_spike_pattern'].str.pad(max_length,side='right',fillchar='0')

            
            df.to_csv(self.dir+f'/{output_fname}_{self.grid}x{self.grid}.csv', columns=['α_fast','β_fast','converted_spike_pattern'], index=False, mode='a', header=False)

 

    def convert_to_num_array(self,spike_pattern,max_length):

        converted_patterns=list()
        for i,pattern in spike_pattern.items():
            num_array=(self.convert_to_symbol(pattern))
            num_array=np.pad(num_array,(0,max_length-len(num_array)),'constant')
            converted_patterns.append(num_array)

        
        return np.array(converted_patterns)

    def get_spikes_as_num_array(self):
        spike_patterns, max_length = self.read_file()
       
        print("length: ",max_length)

        patterns=list()
        for i in range(0,len(spike_patterns)):
            symbol = spike_patterns[i]
            
            num_array=(self.convert_to_symbol(symbol))
            #print(type(num_array))
            if len(num_array) > max_length:
                print("offending one: ", i)

            num_array=np.pad(num_array,(0,max_length-len(num_array)),'constant')

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
dist.handle_pattern()
#print(vec)
#print(dist.get_spikes_as_num_array())
#print(type(dist.get_spikes_as_num_array()[1]))
#arr=dist.get_spikes_as_num_array()[1]
#print(np.pad(arr,(0,90-len(arr)),'constant'))
#dist.get_spikes_as_num_array()
 
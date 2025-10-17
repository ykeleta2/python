import kmedoids
import numpy as np, csv
import matplotlib as plt, matplotlib.pyplot as pyplot


class Clustering:

    def __init__(self):
        self.dir="src/pyloric/data"

    def load_data(self, fname):
        full_name=self.dir+'/'+fname
        #input = np.loadtxt(full_name, dtype='i', delimiter=',')
        with open(full_name) as f:
            lines = (line for line in f if not line.startswith('#'))
            dist_matrix = np.loadtxt(lines, delimiter=',', skiprows=0)

        print(dist_matrix.shape)
        return dist_matrix


    def calculate_clusters(self,dist_matrix):
        dim1=dist_matrix.shape[0]
        dim2=dist_matrix.shape[1]
        print(dim1)
        print(dim2)
        d1=int(round(np.sqrt(dim1)))
        d2=int(round(np.sqrt(dim2)))
        num_clusters=15
        #c = kmedoids.fasterpam(dist_matrix, 10)
        c = kmedoids.pam(dist_matrix,num_clusters,500,"random")
        print(len(c.labels))
        print(c)
        cluster_assignments=c.labels
        assignment_matrix=cluster_assignments.reshape(d1,d2)
        return assignment_matrix 
    
    def plot(self,matrix):
        fig, ax = pyplot.subplots()
        im = ax.imshow(matrix)
        ax.invert_yaxis()
        fig.tight_layout()
        pyplot.show()

    def run(self,file_name):
        dist_matrix=self.load_data(file_name)
        assignment_matrix=self.calculate_clusters(dist_matrix)
        self.plot(assignment_matrix)

    
clustering=Clustering()
#clustering.run("distances_matrix_50.csv")   
#clustering.run("distances_matrix_50_levenshtein.csv")   
clustering.run("output_75x75.csv") 
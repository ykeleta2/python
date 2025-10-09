import numpy as np

def optimal_string_alignment_distance(s1, s2):
    # Create a table to store the results of subproblems
    #dp = [[0 for j in range(len(s2)+1)] for i in range(len(s1)+1)]
    m=len(s1) 
    n=len(s2) 

    v1=np.zeros(n+1)
    v2=np.zeros(n+1)

    for i in range(0,n):
        v1[i]=i
        #print("v1: ",v1)
    # Initialize the table
    # Populate the table using dynamic programming
    for i in range(0, m-1):
       # print("i ",i)
        v2[0]=i+1
      #  print("v2: ",v2)
        for j in range(0, n-1):
           # print("i: ", i, "j: ",j)
            deleteCost=v1[j+1] + 1
            insertCost=v2[j] + 1

            if s1[i] == s2[j]:
                substitutionCost=v1[j]
            else:
                substitutionCost=v1[j]+1

            v2[j+1]=min(deleteCost,insertCost,substitutionCost)
        
        #print("v1: ",v1)
        #print("v2: ",v2)
    v1=v2
    # Return the edit distance
   # print(v1)
    return v1[n]

print(optimal_string_alignment_distance([1,2,2,3,6,0],[1,2,3,3,3,6]))#"geeks", "forgeeks")) g=1,e=2,k=3,f=4,o=5,s=6,r=7
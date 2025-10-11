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

def string_distance(s1,s2):
    m=len(s1)+1 
    n=len(s2)+1 

    v1=[None]*(n)
    v2=[0]*(n)
    
    for i in range(0,n):
        v1[i]=i
    print("m: ",m," n: ",n)
    print("before: v1: ",v1)
    print("before: v2: ",v2)


	# Initialize the table
	# Populate the table using dynamic programming
    for i in range(1, m):
        v2[0]=i
        print("-------------------------------")
        for j in range(1, n):
            #print("j :",j)
            deleteCost=v1[j] + 1
            insertCost=v2[j-1] + 1

            if s1[i-1] == s2[j-1]:
                print(s1[i-1],"==",s2[j-1])
                print("j ",j," V1 is: ",v1, " v2 is: ",v2)
                print(v1)
                substitutionCost=v1[j-1]
                print("deleteCost ",deleteCost," insertCost ",insertCost," substition ",substitutionCost)
            else:
                print(s1[i-1],"!=",s2[j-1])
                print("j ",j," V1 is: ",v1, " v2 is: ",v2)
                substitutionCost=v1[j-1]+1
                print("deleteCost ",deleteCost," insertCost ",insertCost," substition ",substitutionCost)
            v2[j]=min(deleteCost,insertCost,substitutionCost)
            print("V1: ",v1)
            print("V2: ",v2)
        
        v1=v2.copy()
        print("V1 is now: ",v1)
        print("<<<<<<<<<<<<<<<<<<<<<<<<")
    #print(v1," n ",n)
    return v1[n-1]

def string_distance2(s1,s2):
    m=len(s1)+1 
    n=len(s2)+1 

    v1=[None]*(n)
    v2=[0]*(n)
    
    for i in range(0,n):
        v1[i]=i
   # print("m: ",m," n: ",n)
   # print("before: v1: ",v1)
   # print("before: v2: ",v2)

	# Initialize the table
	# Populate the table using dynamic programming
    for i in range(1, m):
        v2[0]=i
       # print("-------------------------------")
        for j in range(1, n):
            #print("j :",j)
            deleteCost=v1[j] + 1
            insertCost=v2[j-1] + 1

            if s1[i-1] == s2[j-1]:
        #        print(s1[i-1],"==",s2[j-1])
        #        print("j ",j," V1 is: ",v1, " v2 is: ",v2)
        #        print(v1)
                substitutionCost=v1[j-1]
        #        print("deleteCost ",deleteCost," insertCost ",insertCost," substition ",substitutionCost)
            else:
           #     print(s1[i-1],"!=",s2[j-1])
           #     print("j ",j," V1 is: ",v1, " v2 is: ",v2)
                substitutionCost=v1[j-1]+1
        #        print("deleteCost ",deleteCost," insertCost ",insertCost," substition ",substitutionCost)
            v2[j]=min(deleteCost,insertCost,substitutionCost)
        #    print("V1: ",v1)
        #    print("V2: ",v2)
        
        v1=v2.copy()
      #  print("V1 is now: ",v1)
      #  print("<<<<<<<<<<<<<<<<<<<<<<<<")
    #print(v1," n ",n)
    return v1[n-1]

#print(optimal_string_alignment_distance([1,2,2,3,6,0],[1,2,3,3,3,6]))#"geeks", "forgeeks")) g=1,e=2,k=3,f=4,o=5,s=6,r=7
s1=[1,4,1,1,1,1,1,3,4,3,3,3,3,3,3,1,1,1,1,1,1,1,2,4,2,2,2,2,2,2,4,1,4,1,1,1,1,1,3,4,3,3,3,3,3,3,1,1,1,1,1,1,1,2,4,2,2,2,2,2,2,4,1,4,1,1,1,1,1,3,4,3,3,3,3,3,3,1,1,1,1,1,1,1,0,0,0,0,0,0]
s2=[1,4,3,1,1,1,1,3,4,3,3,3,3,3,3,1,1,1,1,1,1,1,2,4,2,2,2,2,2,2,4,1,4,1,1,1,1,1,3,4,3,3,3,3,3,3,1,1,1,1,1,1,1,2,4,2,2,2,2,2,2,4,1,4,1,1,1,1,1,3,4,3,3,3,3,3,3,1,1,1,1,1,1,1,0,0,0,0,0,0]
print(string_distance2(s1,s2))
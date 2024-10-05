from distSolution import distance_solution
from angleSolution import angle_solution
import pandas as pd
import numpy as np
from plotNet_2_0 import plotNet3,plotNet2
from distAzimuth import getDist, getAzimuth
from adjustment_2_0 import fixed, free, getD
from updateCoordinates import updateCoordinates


# Data directory
data_dir = "C:\\Users\\Allan\\Documents\\Year 5 Project\\Test Data\\FT"

# Results directory
result_dir = "C:\\Users\\Allan\\Documents\\Year 5 Project\\Results"

# Appoximate coordinates of the controls
df_coord = pd.read_csv(f"{data_dir}\\ApproxCoord.csv")

# Observed angles
df_angles = pd.read_csv("C:/users/Allan/Documents/Year 5 Project/Test Data/FT/Angles.csv")

# Observed distances
#df_distance = pd.read_csv(f"{data_dir}\\Distances.csv")


#A_s, L_s = distance_solution(df_distance, df_coord)
#B, A_a, L_a = angle_solution(df_angles, df_coord)
#np.savetxt("Btest.csv", B, delimiter=",",fmt = '%.9f')

# Making Complete A and L Matrices
# Pad the smaller array with zeros
#a_padded = np.pad(A_s, ((0, 0), (0, A_a.shape[1] - A_s.shape[1])), 'constant', constant_values=0)
# Stack the arrays vertically
#A = np.vstack((a_padded, A_a))
#np.savetxt("Atest.csv", A, delimiter=",",fmt = '%.9f')

# Pad the smaller array with zeros
#l_padded = np.pad(L_s, ((0, 0), (0, L_a.shape[1] - L_s.shape[1])), 'constant', constant_values=0)

# Stack the arrays vertically
#L = np.vstack((l_padded, L_a))
#L = np.float64(L)
#A = np.float64(A)

#np.savetxt("L.csv", L, delimiter=",",fmt = '%.9f')

#X, CovX = free(A,L)
#X2 = fixed(A,L,getD([1,2,7,12],len(A[1,:])))

#new = updateCoordinates(X,f"{data_dir}\\ApproxCoord.csv",result_dir,"deletefree")
#new2 = updateCoordinates(X2,f"{data_dir}\\ApproxCoord.csv",result_dir,"deletefixed")
        
#plotNet(new, CovX )

A = np.array(pd.read_csv("Atest.csv"))
L = np.array(pd.read_csv("L.csv"))


angles_df = pd.read_csv('C:\\Users\\Allan\\Documents\\Year 5 Project\\Test Data\\FT\\Angles.csv')

## Adjustments
C, cX, Var, Cov, eValues, eVectors = free(A,L,df_coord)
print(f'''
      Vector: {eVectors.shape}
      Values: {eValues.shape}
      Cov: {Cov.shape}
      ''')
#plotNet2(df_coord, angles_df, Var, Cov,eValues,eVectors,"FREE_ADJUSTMENT")

#cX.to_csv("Free Model Estimates.csv")
#C.to_csv("Free Model Coordinates.csv")

C, cX, Var, Cov, eValues, eVectors =fixed(A,L,getD([1,12],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov,eValues,eVectors,[1,12],"FIXED_NETWORK_1_12")

#cX.to_csv("Fixed Model Estimates 1, 12 Fix.csv")
#C.to_csv("Fixed Model Coordinates 1, 12 Fix.csv")

C, cX, Var, Cov, eValues, eVectors =fixed(A,L,getD([2,3,5,10],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov,eValues, eVectors,[2,3,5,10],"FIXED_NETWORK_2_3_5_10")

#cX.to_csv("Fixed Model Estimates 2, 3, 5, 10 Fix.csv")
#C.to_csv("Fixed Model Coordinates 2, 3, 5, 10 Fix.csv")

C, cX, Var, Cov, eValues, eVectors =fixed(A,L,getD([4,5,12],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov,eValues, eVectors,[2,3,5,10],"FIXED_NETWORK_4_5_12")

#cX.to_csv("Fixed Model Estimates 4, 5, 12 Fix.csv")
#C.to_csv("Fixed Model Coordinates 4, 5, 12 Fix.csv")

C, cX, Var, Cov, eValues, eVectors =fixed(A,L,getD([5,6,8],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov, eValues, eVectors,[5,6,8], "FIXED_NETWORK_5_6_8")

#cX.to_csv("Fixed Model Estimates 5, 6, 8 Fix.csv")
#C.to_csv("Fixed Model Coordinates 5, 6, 8 Fix.csv")


C, cX, Var, Cov, eValues, eVectors=fixed(A,L,getD([5,8,10,12],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov, eValues, eVectors, [5,8,10,12],"FIXED_NETWORK_5_8_10_12")

#cX.to_csv("Fixed Model Estimates 5, 8, 10, 12 Fix.csv")
#C.to_csv("Fixed Model Coordinates 5, 8, 10, 12 Fix.csv")

C, cX, Var, Cov, eValues, eVectors =fixed(A,L,getD([5,10],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov,eValues, eVectors,[5,10],"FIXED_NETWORK_5_10")

#cX.to_csv("Fixed Model Estimates 5, 10 Fix.csv")
#C.to_csv("Fixed Model Coordinates 5, 10 Fix.csv")

C, cX, Var, Cov,eValues, eVectors =fixed(A,L,getD([9,10,11],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov,eValues, eVectors,[9,10,11], "FIXED_NETWORK_9_10_11")

#cX.to_csv("Fixed Model Estimates 9, 10, 11 Fix.csv")
#C.to_csv("Fixed Model Coordinates 9, 10, 11 Fix.csv")

C, cX, Var, Cove,Values, eVectors =fixed(A,L,getD([1,2,3,4],len(A[1,:])),df_coord)
plotNet3(df_coord, angles_df, Var, Cov,eValues, eVectors,[1,2,3,4],"FIXED_NETWORK_1_2_3_4")

#cX.to_csv("Fixed Model Estimates 1, 2, 3, 4 Fix.csv")
#C.to_csv("Fixed Model Coordinates 1, 2, 3, 4 Fix.csv")


#np.savetxt("Free Model Estimates.csv", cX, delimiter=",",fmt = '%.9f')
#np.savetxt("Fixed Model Estimates Itn.txt", iteration)

#np.savetxt("Fixed Model Estimates 12 Fix.csv", cX, delimiter=",",fmt = '%.9f')
#np.savetxt("Fixed Model Estimates 1 Fix Itn.csv", iteration, delimiter=",",fmt = '%.9f')
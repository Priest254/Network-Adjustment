import numpy as np
import pandas as pd
from distAzimuth import getAzimuth, getDist
def distance_solution(df_distance, df_coord):

    # Computing approximate distances using approximate coordinates for the observed baselines
    df_merged = df_distance.merge(df_coord, left_on='FROM', right_on='STATION') \
                               .merge(df_coord, left_on='TO', right_on='STATION', suffixes=('_FROM', '_TO'))

    df_merged['APPROX_DISTANCE'] = df_merged.apply(lambda row: getDist(row), axis=1)

    df_distance['APPROX'] = df_merged['APPROX_DISTANCE']


    # Initialize an empty list to store A
    A = [] # A matrix of observation for thE distances
    
    # Iterate over the rows of the df_distance DataFrame
    for index, row in df_distance.iterrows():
        # Fetch the 'FROM' and 'TO' points
        from_point = row['FROM']
        to_point = row['TO']
        
        # Fetch the corresponding coordinates from the df_coord DataFrame
        xi, yi = df_coord[df_coord['STATION'] == from_point][['X', 'Y']].values[0]
        xj, yj = df_coord[df_coord['STATION'] == to_point][['X', 'Y']].values[0]
        
        # approx distance IJ 
        IJ = np.float64(row['APPROX'])

        # append A
        A.append([(xi - xj)/IJ, (yi - yj)/IJ, (xj - xi)/IJ, (yj - yi)/IJ])
        
    A = np.float64(np.array(pd.DataFrame(A)))

    # Define the desired 6x8 pattern 'B'
    B_pattern = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ])

    # Initialize a 6x8 array 'B' with zeros
    B = np.zeros((6, 8))

    # Create a mapping from the indices in 'A' to the indices in 'B'
    mapping = np.where(B_pattern == 1)

    # Rearrange the elements in 'A' to match the pattern in 'B'
    B[mapping] = A.flatten()
    A = np.float64(B)
    L = np.float64(np.array(df_distance['DISTANCE'] - df_distance['APPROX']).reshape(6,1))
    
    return A, L

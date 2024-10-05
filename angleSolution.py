from distAzimuth import getDist2, getDist, getAzimuth
import numpy as np
import pandas as pd
import math as m

def angle_solution(df_angles, df_coord):
    """
    Computes the design matrix A and observation vector L for observed angles.
    
    Parameters:
    df_angles (pd.DataFrame): DataFrame containing observed angles with columns ['FROM', 'TO', 'THROUGH', 'RADS'].
    df_coord (pd.DataFrame): DataFrame containing coordinates with columns ['STATION', 'X', 'Y'].
    
    Returns:
    A (np.ndarray): Design matrix.
    L (np.ndarray): Observation vector.
    """
    # Merging the dataframes to compute approximate distances
    df_merged = df_angles.merge(df_coord, left_on='FROM', right_on='STATION') \
                         .merge(df_coord, left_on='TO', right_on='STATION', suffixes=('_FROM', '_TO'))

    # Compute approximate distances
    df_merged['APPROX_DISTANCE'] = df_merged.apply(lambda row: getDist(row), axis=1)
    df_angles['APPROX'] = df_merged['APPROX_DISTANCE']

    # Initialize lists to store the design matrix (A) and observation vector (L)
    A = []
    L = []
    angle = []  # Computed angles
    obs_angle = df_angles['RADS']  # Observed angles

    # Iterate over the rows of df_angles
    for index, row in df_angles.iterrows():
        station = row['FROM']
        fore = row['TO']
        back = row['THROUGH']
        #print(f'Station: {station} Backsight: {back} Foresight: {fore}')

        xi, yi = df_coord[df_coord['STATION'] == station][['Y', 'X']].values[0]
        xb, yb = df_coord[df_coord['STATION'] == back][['Y', 'X']].values[0]
        xf, yf = df_coord[df_coord['STATION'] == fore][['Y', 'X']].values[0]

        # Compute azimuths and intermediate angles
        Foresight = getAzimuth(xi, yi, xf, yf)
        Backsight = getAzimuth(xi, yi, xb, yb)

        if Foresight >= Backsight:
            i_angle = Foresight - Backsight
        else:
            i_angle = Foresight - Backsight + 360

        angle.append(m.radians(i_angle))

        # Compute distances squared
        IB = m.pow(getDist2(xi, yi, xb, yb), 2)
        IF = m.pow(getDist2(xi, yi, xf, yf), 2)

        # Append computed values to the design matrix A
        A.append([
            (yi - yb) / IB, (xb - xi) / IB, ((yb - yi) / IB) - ((yf - yi) / IF),
            ((xi - xb) / IB - (xi - xf) / IF), (yf - yi) / IF, (xi - xf) / IF
        ])

    A = np.array(pd.DataFrame(A))

    # Define the desired 6x8 pattern 'B'
    B_pattern = np.array(pd.read_csv('pattern2.csv'))

    # Initialize a 48x24 array 'B' with zeros
    B = np.zeros((44, 24))
    
    # Create a mapping from the indices in 'A' to the indices in 'B'
    mapping = np.where(B_pattern == 1)

    # Rearrange the elements in 'A' to match the pattern in 'B'
    B[mapping] = A.flatten()
    A = B

    # Compute the observation vector L
    L = np.float64(obs_angle - angle)
    L = np.array(L).reshape(44, 1)

    return A, L

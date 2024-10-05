# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:30:52 2024

@author: Allan
"""
import math as m
from distAzimuth import getDist
#  A formation


def get_A_matrix_elements(angles_df, coords_df):
    """
    Computes the design matrix A, for observed distances
    Takes the observed angles, and initial coordinates as input
    """
   A = []

  # Computing approximate distances using approximate coordinates for the observed baselines
  df_merged = df_distance.merge(df_coord, left_on='FROM', right_on='STATION') \
                             .merge(df_coord, left_on='TO', right_on='STATION', suffixes=('_FROM', '_TO'))

  df_merged['APPROX_DISTANCE'] = df_merged.apply(lambda row: getDist(row), axis=1)

  df_distance['APPROX'] = df_merged['APPROX_DISTANCE']

  # Iterate over each row in the angles dataframe
  for index, row in angles_df.iterrows():
    # Extract station names
    from_station = row['From']
    through_station = row['Through']
    to_station = row['To']

    # Get corresponding coordinates from the coordinates dataframe
    station = coords_df[coords_df['Station'] == from_station][['Y', 'X']].values[0]
    back = coords_df[coords_df['Station'] == through_station][['Y', 'X']].values[0]
    fore = coords_df[coords_df['Station'] == to_station][['Y', 'X']].values[0]

    yi, xi = station
    yf, xf = fore
    yb, xb = back

    IB = m.pow(getDist(xi, yi, xb, yb),2)
    IF = m.pow(getDist(xi, yi, xf, yf),2)

    A.append([(yi - yb) / IB, (xb - xi) / IB, ((yb - yi) / IB) - ((yf - yi) / IF),
              ((xi - xb)/IB - (xi - xf)/IF), (yf - yi)/IF, (xi - xf)/IF])

    L = np.float64(np.array(angles_df['DISTANCE'] - df_distance['APPROX']).reshape(6,1))
  return A, L

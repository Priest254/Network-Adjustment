# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:30:55 2024

@author: Allan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib_scalebar.scalebar import ScaleBar

# Read the CSV files
coords_df = pd.read_csv('ApproxCoord.csv')
angles_df = pd.read_csv('C:\\Users\\Allan\\Documents\\Year 5 Project\\Test Data\\FT\\Angles.csv')

def plotNet(coords_df, angles_df, StD):
    """
    Plot the geodetic network with error ellipses.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates with columns ['STATION', 'X', 'Y'].
    angles_df (pd.DataFrame): DataFrame containing angle data with columns ['FROM', 'TO'].
    StD (np.ndarray): Standard deviations reshaped to (12, 2).
    """
    std_reshaped = StD.reshape(-1, 2)
    
    plt.figure(figsize=(16, 12), dpi=720)
    plt.grid(color='blue', linestyle='-', linewidth=0.1)
    
    plt.scatter(coords_df['Y'], coords_df['X'], color='blue')
    for i, row in coords_df.iterrows():
        plt.text(row['Y'], row['X'], row['STATION'], fontsize=12, ha='right', va='bottom')
    
    for _, row in angles_df.iterrows():
        from_station = row['FROM']
        to_station = row['TO']
        
        from_coords = coords_df[coords_df['STATION'] == from_station].iloc[0]
        to_coords = coords_df[coords_df['STATION'] == to_station].iloc[0]
        
        x_coords = [from_coords['Y'], to_coords['Y']]
        y_coords = [from_coords['X'], to_coords['X']]
        plt.plot(x_coords, y_coords, 'green')
    
    plt.xlabel('EASTINGS')
    plt.ylabel('NORTHINGS')
    plt.title('DESIGN NETWORK')
    plt.savefig("DESIGN_NETWORK.jpg", format='jpeg', dpi=700)
    plt.show()

def plotNet2(coords_df, angles_df, Var, Cov,eValues,eVectors, name):
    """
    Plot the geodetic network with error ellipses and covariance information.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates with columns ['STATION', 'X', 'Y'].
    angles_df (pd.DataFrame): DataFrame containing angle data with columns ['FROM', 'TO'].
    Var (np.ndarray): Variance of the estimates.
    Cov (np.ndarray): Covariance values.
    name (str): Title of the plot and name of the output file.
    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=720)
    ax.grid(color='blue', linestyle='-', linewidth=0.1)
    ax.scatter(coords_df['Y'], coords_df['X'], color='green')
    

    
    for i, row in coords_df.iterrows():
        ax.text(row['Y'], row['X'], row['STATION'], fontsize=18, ha='right', va='bottom')
        # Filter the data for the current station
        x_center = row['Y']
        y_center = row['X']
        
        # Get the eigenvalues for the current subspace
        lambda1 = eValues[i].real
        lambda2 = eValues[i+1].real
        
        # Get the eigenvectors for the current subspace
        vector1 = eVectors[:, i].real
        vector2 = eVectors[:, i+1].real

        # Compute the angle of the ellipse
        angle = np.degrees(np.arctan2(vector2[1], vector2[0]))
        
        # Calculate the lengths of the semi-major and semi-minor axes
        width, height = 2 * 0.001 * np.sqrt(lambda1)*25, 2 * 0.001 * np.sqrt(lambda2)*25
        print(width,height)
        # Create the ellipse
        ellipse = Ellipse(xy=(x_center, y_center), width=width, height=height, angle=angle,
                          edgecolor='red', fc='None', lw=2)
        ax.add_patch(ellipse)
    
    '''for i, row in coords_df.iterrows():
        Vx = Var[i, 0]
        Vy = Var[i, 1]
        xy = Cov[i]
        
        w = np.sqrt((Vx - Vy)**2 + (4*xy)**2)
        a = np.sqrt(0.5 * (Vx + Vy + w))
        b = np.sqrt(0.5 * (Vx + Vy - w))
        
        ellipse = Ellipse((row['Y'], row['X']), width=a*0.1, height=b*0.1, edgecolor='red', fc='None', lw=2)
        ax.add_patch(ellipse)'''
 

    
    for _, row in angles_df.iterrows():
        from_station = row['FROM']
        to_station = row['TO']
        
        from_coords = coords_df[coords_df['STATION'] == from_station].iloc[0]
        to_coords = coords_df[coords_df['STATION'] == to_station].iloc[0]
        
        x_coords = [from_coords['Y'], to_coords['Y']]
        y_coords = [from_coords['X'], to_coords['X']]
        ax.plot(x_coords, y_coords, 'blue')
        
     # Add a network scale bar
    network_scalebar = ScaleBar(1, units='m', location='lower left', length_fraction=0.2,
                                scale_loc='bottom', label='Network Scale', dimension='si-length', color='black')
    ax.add_artist(network_scalebar)

    # Add an ellipse scale bar
    ellipse_scalebar = ScaleBar(0.001, units='m', location='lower right', length_fraction=0.2,
                                scale_loc='bottom', label='Ellipse Scale', dimension='si-length', color='red')
    ax.add_artist(ellipse_scalebar)

    ax.set_xlabel('EASTINGS')
    ax.set_ylabel('NORTHINGS')
    ax.set_title(name)
    
    plt.savefig(f'{name}.jpg', format='jpeg', dpi=700)
    plt.show()

# Example usage:
# plotNet(coords_df, angles_df, StD)
# plotNet2(coords_df, angles_df, Var, Cov, "Network with Ellipses")

def plotNet3(coords_df, angles_df, Var, Cov,eValues,eVectors,st, name):
    """
    Plot the geodetic network with error ellipses and covariance information.

    Parameters:
    coords_df (pd.DataFrame): DataFrame containing coordinates with columns ['STATION', 'X', 'Y'].
    angles_df (pd.DataFrame): DataFrame containing angle data with columns ['FROM', 'TO'].
    Var (np.ndarray): Variance of the estimates.
    Cov (np.ndarray): Covariance values.
    name (str): Title of the plot and name of the output file.
    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=720)
    ax.grid(color='blue', linestyle='-', linewidth=0.1)
    
    for i, row in coords_df.iterrows():
        if row["STATION"] in st:
            ax.scatter(row['Y'], row['X'], marker='^',s=300, color='red')
        else:
            ax.scatter(row['Y'], row['X'], color='green')
    
    for i, row in coords_df.iterrows():
        if row["STATION"] in st:
            print(row["STATION"])
            ax.text(row['Y'], row['X'], row['STATION'], fontsize=18, ha='right', va='bottom')
            
            pass
        else:
            
            ax.text(row['Y'], row['X'], row['STATION'], fontsize=18, ha='right', va='bottom')
            # Filter the data for the current station
            x_center = row['Y']
            y_center = row['X']
            
            # Get the eigenvalues for the current subspace
            lambda1 = eValues[i]
            lambda2 = eValues[i+1]
            
            # Get the eigenvectors for the current subspace
            vector1 = np.array(eVectors[:, i])
            vector2 = np.array(eVectors[:, i+1])
            
            # Compute the angle of the ellipse
            print(f'v1:: {lambda1} v2:: {lambda2}')
            angle = np.degrees(np.arctan2(vector2[1], vector2[0]))
            
            # Calculate the lengths of the semi-major and semi-minor axes
            width, height = 2 * 0.2 * np.sqrt(lambda1), 2 * 0.2 * np.sqrt(lambda2)
            
            # Create the ellipse
            ellipse = Ellipse(xy=(x_center, y_center), width=width, height=height, angle=angle,
                              edgecolor='red', fc='None', lw=2)
            ax.add_patch(ellipse)
    
    
    for _, row in angles_df.iterrows():
        from_station = row['FROM']
        to_station = row['TO']
        
        from_coords = coords_df[coords_df['STATION'] == from_station].iloc[0]
        to_coords = coords_df[coords_df['STATION'] == to_station].iloc[0]
        
        x_coords = [from_coords['Y'], to_coords['Y']]
        y_coords = [from_coords['X'], to_coords['X']]
        ax.plot(x_coords, y_coords, 'blue')
    
    
    
     # Add a network scale bar
    network_scalebar = ScaleBar(1, units='m', location='lower left', length_fraction=0.2,
                                scale_loc='bottom', label='Network Scale', dimension='si-length', color='black')
    ax.add_artist(network_scalebar)

    # Add an ellipse scale bar
    ellipse_scalebar = ScaleBar(0.001, units='m', location='lower right', length_fraction=0.2,
                                scale_loc='bottom', label='Ellipse Scale', dimension='si-length', color='red')
    ax.add_artist(ellipse_scalebar)
    #
    ax.set_xlabel('EASTINGS')
    ax.set_ylabel('NORTHINGS')
    ax.set_title(name)
    

    
    plt.savefig(f'{name}.jpg', format='jpeg', dpi=700)
    plt.show()

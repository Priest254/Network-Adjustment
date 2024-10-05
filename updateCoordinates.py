import numpy as np
import pandas as pd

def updateCoordinates(cX,path_to_original,path_to_new,file_name):
    ''' Get New Coordinates '''
    cX = cX.reshape(-1,2)
    
    new = pd.read_csv(path_to_original)
    print(f'Original Coordinates: \n{new}')
    x = np.array(new['X']).reshape(12,1)
    y = np.array(new['Y']).reshape(12,1)
    x += cX[:,0].reshape(12,1)
    y += cX[:,1].reshape(12,1)
    
    new['X'] = x
    new['Y'] = y
    

    # Round the 'X' and 'Y' columns to 3 decimal places
    new['X'] = new['X'].round(3)
    new['Y'] = new['Y'].round(3)
    print(f'New Coordinates: \n{new}')

    new.to_csv(f'{path_to_new}\\Coordinates\\{file_name}.csv')
    return new
    

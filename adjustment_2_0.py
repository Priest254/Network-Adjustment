import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def free(A, L, df):
    """
    Performs free adjustment of a geodetic network.
    
    Parameters:
    A (numpy.ndarray): Design matrix.
    L (numpy.ndarray): Observation vector.
    df (pandas.DataFrame): Initial coordinates with columns ['Station', 'X', 'Y'].

    Returns
    Updated_Coords (pandas.DataFrame): Updated coordinates after adjustment.
    Estimates (pandas.DataFrame): Adjustment estimates.
    Var (numpy.ndarray): Variance of the estimates.
    Cov (numpy.ndarray): Covariance values.
    """
    # Initial computations
    W = np.eye(len(A))
    N = A.T @ W @ A  # Normal equation matrix
    Qxx = np.linalg.pinv(N)  # Cofactor matrix
    M = A.T @ W @ L  # Absolute vector matrix

    # Least square solution
    X, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(Qxx, M, damp=10, calc_var=True)
    X = X.reshape((len(A[1,:]),1))
    
    # Prepare initial coordinates
    x_values = df['X'].values
    y_values = df['Y'].values
    Coords = np.empty((len(A[0]), 1))
    Coords[0::2] = x_values.reshape(-1, 1)
    Coords[1::2] = y_values.reshape(-1, 1)
    Coords = Coords.reshape(-1, 2)
    
    # Calculate residuals and covariance matrix
    V = L - (A @ X)
    n, p = A.shape
    So2 = np.mean((V.T @ W @ V) / (n - p))
    CovX = So2 * Qxx
    
    Var = np.diagonal(CovX).reshape(-1, 2)
    abD = CovX[np.arange(len(CovX) - 1), np.arange(1, len(CovX))]
    Cov = abD[::2].reshape(-1, 1)
    eValues = np.linalg.eig(CovX)[0].reshape(24,1)
    eVectors = np.linalg.eig(CovX)[1]
    
    # Updated coordinates
    X = X.reshape(-1, 2)
    Updated_Coords = Coords + X

    # Create the dictionary to hold Estimates
    Estimates = {
        'Station': [i + 1 for i in range(X.shape[0])],
        'cX': X[:, 0],
        'cY': X[:, 1]
    }
    
    # Create the dictionary to hold updated coordinates
    Updated_Coords = {
        'Station': [i + 1 for i in range(X.shape[0])],
        'X': Updated_Coords[:, 0],
        'Y': Updated_Coords[:, 1]
    }
    
    # Convert the dictionary to a pandas DataFrame
    Estimates = pd.DataFrame(Estimates)
    Updated_Coords = pd.DataFrame(Updated_Coords)
    
    return Updated_Coords, Estimates, Var, Cov, eValues, eVectors

def fixed(A, L, D, df):
    """
    Performs fixed adjustment of a geodetic network with constraints.
    
    Parameters:
    A (numpy.ndarray): Design matrix.
    L (numpy.ndarray): Observation vector.
    D (numpy.ndarray): Constraint matrix.
    df (pandas.DataFrame): Initial coordinates with columns ['Station', 'X', 'Y'].

    Returns:
    Updated_Coords (pandas.DataFrame): Updated coordinates after adjustment.
    Estimates (pandas.DataFrame): Adjustment estimates.
    Var (numpy.ndarray): Variance of the estimates.
    Cov (numpy.ndarray): Covariance values.
    """
    # Initial computations
    W = np.eye(len(A))
    N = A.T @ W @ A  # Normal equation matrix
    M = A.T @ W @ L  # Absolute vector matrix

    # Helmerts Constraint Equation
    row1 = np.hstack((N, D.T))
    row2 = np.hstack((D,np.zeros((len(D[:,1]),len(D[:,1])))))
    
    B = np.vstack((row1, row2))
    Bxx = np.linalg.inv(B)
    Ll = np.vstack((M, np.zeros((len(D), 1))))

    # Least square solution
    X, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(Bxx, Ll, damp=10, calc_var=True)
    X = X.reshape(len(B[1,:]),1)
    V = Ll - (B @ X)
    X = np.round(X[:len(A[0])], 9).reshape(len(A[1,:]),1)
    print(np.round(arnorm,6))
    
    # Check and adjust the solution
    checkX = np.round((Bxx @ Ll), 9)[:len(A[0])]
    X[checkX == 0] = 0
    var = np.round(var[:len(A[0])], 9).reshape(-1, 2)
    
    # Prepare initial coordinates
    x_values = df['X'].values
    y_values = df['Y'].values
    Coords = np.empty((len(A[0]), 1))
    Coords[0::2] = x_values.reshape(-1, 1)
    Coords[1::2] = y_values.reshape(-1, 1)
    Coords = Coords.reshape(-1, 2)

    # Calculate residuals and covariance matrix
    
    r = len(A) - len(A[0])
    print(V.shape)
    So2 = (V.T  @ V) / r
    CovX = So2 * Bxx
    CovX = (CovX + CovX.T) / 2
    threshold = 1e-9
    CovX[np.abs(CovX) < threshold] = 0

    Var = np.diagonal(CovX)[:len(A[0])].reshape(-1, 2)
    abD = CovX[np.arange(len(A[0]) - 1), np.arange(1, len(A[0]))]
    Cov = abD[::2].reshape(-1, 1)
    
    eValues = np.linalg.eig(CovX)[0]
    
    eVectors = np.linalg.eig(CovX)[1]
    print(eValues)

    # Updated coordinates
    X = X.reshape(-1, 2)
    Updated_Coords = Coords + X

    # Create the dictionary to hold Estimates
    Estimates = {
        'Station': [i + 1 for i in range(X.shape[0])],
        'cX': X[:, 0],
        'cY': X[:, 1]
    }
    
    # Create the dictionary to hold updated coordinates
    Updated_Coords = {
        'Station': [i + 1 for i in range(X.shape[0])],
        'X': Updated_Coords[:, 0],
        'Y': Updated_Coords[:, 1]
    }
    
    # Convert the dictionary to a pandas DataFrame
    Estimates = pd.DataFrame(Estimates)
    Updated_Coords = pd.DataFrame(Updated_Coords)
    
    return Updated_Coords, Estimates, Var, Cov, eValues, eVectors


def getD(m,n):
    """
    Computes the 2D constraint matrix for fixed stations.
    
    Parameters:
    m (list): List of fixed stations.
    n (int): Number of unknowns.
    
    Returns:
    numpy.ndarray: Constraint matrix.
    """
    D = []
    columns = len(m) * 2
    b = np.zeros((columns,n))
    #b = np.list(b)
    for i in range(0,len(m)):
        station = m[i]
        b[i+i,station-1+station-1] = 1
        b[i+i+1,station+station-1] = 1
    D.append(b)
    return np.array(D)[0]

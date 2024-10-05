import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.stats import chi2
def free(A,cL, df):
    W = np.eye(len(A[:,1]))
    N = A.T @ W @ A                    # Normal equation matrix
    Qxx = np.linalg.pinv(N)               # Cofactor matrix
    M = A.T @ W @ cL                     # Absolute vector matrix

    # Tolerances
    #atol = 0
    #btol = 0
    damp  = 0.000000001 # regularize
    
    # Maximum number of iterations
    iter_lim = 1000

    # Solve for the least squares solution
    cX, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(Qxx, M, #damp=damp,
                                                                       #atol=atol,
                                                                       #btol=btol,
                                                                       iter_lim=iter_lim,
                                                                       show=False,
                                                                       calc_var=True)


    cX = cX[0:len(A[1,:])]
    cX = np.round(cX,6)
    
    # Extract elements from 'X' and 'Y' columns
    x_values = df['X'].values
    y_values = df['Y'].values
    
    # Hold them ready for addition to cX
    Coords = np.empty((len(A[0,:],), 1))
    
    
    # Interleave elements from 'X' and 'Y' arrays into the combined array
    Coords[0::2] = x_values.reshape(-1, 1)
    Coords[1::2] = y_values.reshape(-1, 1)
    
    V = cL - ( A @ cX)
    r = len(A[:,1]) - len(A[1,:])                     # Redundancy
    So2 = np.mean((V.T @ W @ V) /r  )                          # a posteriori variance 
    CovX = So2*Qxx                                    # Covariance matrix
    Var = np.diagonal(CovX).reshape((len(A[1,:]),1))  # Etract variance
    StD = np.sqrt(Var)                                # Standard deviation
    
    # Step 2: Square the residuals
    V2 = V**2

    # Step 3: Sum the squared residuals
    chi_square_stat = np.sum(V2)
    p_value = 1 - chi2.cdf(chi_square_stat, 11)
  
    for xx in cX:
        print(xx)
    print(itn)
    return cX, itn

def fixed(A,L,D):
    np.float64(A),np.float64(L), np.float64(D)
    
    W = np.eye(len(A[:,1]))
    N = np.dot(np.dot(A.T,W),A)                                 # Normal equation matrix
    Qxx = np.linalg.pinv(N)                                     # Cofactor matrix
    M = np.dot(np.dot(A.T,W),L)                                 # Absolute vector matrix
    
    row1 = np.hstack((Qxx,D.T))
    row2 = np.hstack((D,np.zeros((len(D[:,1]),len(D[:,1])))))
    
    B = np.vstack((row1,row2))
    Bxx = np.linalg.pinv(B)
    Ll = np.vstack((M,np.zeros((len(D[:,1]),1))))
    
    # Tolerances
    #atol = 1e+10
    #btol = 1e+10
    damp  = 0 # regularize

    # Maximum number of iterations
    iter_lim = 1000
    
    # Solve for the least squares solution
    X_check, istop, itn2, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(B, Ll,
                                                                       iter_lim=iter_lim,
                                                                       show=False)
    X_check = X_check[0:len(A[1,:])]
    X_check = np.round(X_check,6)
    X_check = abs(X_check)
    
    # Solve for the least squares solution
    X, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(B, Ll,
                                                                       iter_lim=iter_lim,
                                                                       #atol = atol,
                                                                       #btol = btol,
                                                                       #damp = damp,
                                                                       show=False,
                                                                       calc_var=True)
    X = X[0:len(A[1,:])]
    
    XXX  = Bxx @ Ll
    XXX = np.round(XXX[0:len(A[1,:])],9)
    
    # Print the results
    print(f"Number of iterations (itn): {itn2}")
    print(f"Estimated diagonals of the inverse of A^TA(variance): {var.shape}")
    
    # Replace elements of fixed stations
    X = np.where(X_check == 0, 0, X)
    X = np.round(X,6)
    
    for xx in XXX:
        print(xx)
        
    print(itn)
    return X, itn

def getD(m,n):
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
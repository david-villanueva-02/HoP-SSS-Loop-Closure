import numpy as np
from scipy.linalg import lstsq

def compute_homography(CL1uv, CL2uv, model):
    """Estimate the Homography between two images according to model.

    Args:
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Euclidean', 'Similarity', 'Affine', 'Projective'.

    Returns:
        numpy.ndarray: Estimated Homography of Model type. 3x3 matrix.
    """

    CL1uv = np.asarray(CL1uv, dtype=float)
    CL2uv = np.asarray(CL2uv, dtype=float)

    # Homogeneous coordinates
    N = CL1uv.shape[0]

    # Calculate the matrices for coordinate normalisation  
    if model == "Affine" or model == "Projective":

        mean1u = np.mean(CL1uv[:,0])
        std1u = np.std(CL1uv[:,0])
        mean1v = np.mean(CL1uv[:,1])
        std1v = np.std(CL1uv[:,1])
        
        T1 = np.array([[1/(std1u+1e-6), 0, -mean1u/(std1u+1e-6)],
                        [0, 1/(std1v+1e-6), -mean1v/(std1v+1e-6)],
                        [0,0,1]])
        
        mean2u = np.mean(CL2uv[:,0])
        std2u = np.std(CL2uv[:,0])
        mean2v = np.mean(CL2uv[:,1])
        std2v = np.std(CL2uv[:,1])

        T2 = np.array([[1/(std2u+1e-6), 0, -mean2u/(std2u+1e-6)],
                        [0, 1/(std2v+1e-6), -mean2v/(std2v+1e-6)],
                        [0,0,1]])
        
        CL1uv_hom = np.hstack([CL1uv, np.ones((N,1))]) 
        CL2uv_hom = np.hstack([CL2uv, np.ones((N,1))]) 

        # Normalization
        CL1uv = (T1 @  CL1uv_hom.T).T
        CL2uv = (T2 @ CL2uv_hom.T).T

    else:
        T1 = np.eye(3)
        T2 = np.eye(3)

    if model == "Translation":
        
        # Create Q matrix 
        Q = np.eye(2)
        for i in range(N-1):
            Q = np.vstack([Q, np.eye(2)])
            
        # Create b vector
        b = np.zeros((2*N, 1)) # Prealocate the vector
        for i in range(0, 2*N, 2):
            b[i, 0] = CL2uv[i//2, 0] - CL1uv[i//2, 0]
            b[i+1, 0] = CL2uv[i//2, 1] - CL1uv[i//2, 1]

        # Calculates the homography
        x,_,_,_ = lstsq(Q, b)

        # Builds the homography
        (tx, ty) = np.squeeze(x)
        A = np.eye(3)
        A[:2,2] = [tx, ty]

    elif model == "Euclidean":

        # Calculate the centroids
        CL1uv_centroid = np.mean(CL1uv[:,0:2], axis=0)
        CL2uv_centroid = np.mean(CL2uv[:,0:2], axis=0)

        # Center the coordinates
        X = CL1uv[:,0:2] - CL1uv_centroid
        Y = CL2uv[:,0:2] - CL2uv_centroid

        # Calculate the rotation with the orthogonal Procrustes solution
        S = X.T @ Y
        U, _, Vt = np.linalg.svd(S)

        R = Vt.T @ U.T

        # Avoid reflections
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Calculate translation
        t = CL2uv_centroid - R @ CL1uv_centroid

        # Builds the homography
        A = np.eye(3)
        A[:2,:2] = R
        A[:2,2] = t

    elif model == "Similarity":
        
        # Create Q matrix 
        Q = np.zeros((2*N,4))
        for i, j in zip(range(0, 2*N, 2), range(N)):
            u, v = CL1uv[j, 0], CL1uv[j, 1]

            Q[i,:] = np.array([u, -v, 1, 0])
            Q[i+1,:] = np.array([v, u, 0, 1])

        # Create b vector
        b = np.zeros((2*N, 1))
        for i in range(N):
            b[2*i, 0] = CL2uv[i, 0]
            b[2*i+1, 0] = CL2uv[i, 1]

        # Calculate the homography
        x,_,_,_ = lstsq(Q, b)

        # Builds the homography
        (a, b, c, d) = np.squeeze(x)
        A = np.matrix([[a, -b, c],
                       [b, a, d],
                       [0, 0, 1]])

    elif model == "Affine":
        
        # Create Q matrix 
        Q = np.zeros((2*N,6)) # Prealocate matrix
        for i, j in zip(range(0, 2*N, 2), range(N)):
            Q[i,:] = np.array([CL1uv[j,0], CL1uv[j,1], 1, 0, 0, 0])
            Q[i+1,:] = np.array([0, 0, 0, CL1uv[j,0], CL1uv[j,1], 1])    

        # Create b vector
        b = np.zeros((2*N, 1))
        for i in range(N):
            b[2*i, 0] = CL2uv[i, 0]
            b[2*i+1, 0] = CL2uv[i, 1]    

        # Calculate the homography
        x,_,_,_ = lstsq(Q, b)

        # Builds the homography
        (a, b, c, d, e, f) = np.squeeze(x)
        A = np.matrix([[a, b, c],
                       [d, e, f],
                       [0, 0, 1]])

    elif model == "Projective":
        
        # Create Q matrix 
        Q = np.zeros((2*N,8))
        for i, j in zip(range(0, 2*N, 2), range(N)):
            u, v = CL1uv[j, 0], CL1uv[j, 1]
            up, vp = CL2uv[j, 0], CL2uv[j, 1]

            Q[i, :]   = np.array([u, v, 1, 0, 0, 0, -u*up, -v*up])  # u'
            Q[i+1, :] = np.array([0, 0, 0, u, v, 1, -u*vp, -v*vp])  # v'

        # Create b vector
        b = np.zeros((2*N, 1))
        for i in range(N):
            b[2*i, 0] = CL2uv[i, 0]
            b[2*i+1, 0] = CL2uv[i, 1]      

        # Calculate the homography
        x,_,_,_ = lstsq(Q, b)

        # Builds the homograpy
        (a, b, c, d, e, f, g, h) = np.squeeze(x)
        A = np.matrix([[a, b, c],
                       [d, e, f],
                       [g, h, 1]])        

    else:
        print("Invalid model, returning identity homography")
        return np.eye(3)

    # Un-normalise Homography 
    H12 = np.linalg.inv(T2) @ A @ T1

    if abs(H12[2,2]) > 1e-12:
        H12 = H12 / H12[2,2]

    return np.asarray(H12)
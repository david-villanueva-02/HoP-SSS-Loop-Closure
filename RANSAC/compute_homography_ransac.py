import numpy as np 
from math import log 

from .compute_homography import compute_homography
from .projection_error import projection_error

def compute_homography_ransac(CL1uv: np.ndarray, CL2uv: np.ndarray, model: str, 
                              num_iterations: int = None, outlier_percent: float = 0.5, p: float = 0.98, t: int = 10):
    """Estimate the Homography between two images according to model using the RANSAC algorithm.

    Args:
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Similarity', 'Affine', 'Projective'.
        num_iterations (int, optional): Number of iterations to run RANSAC. Defaults to None.
        outlier_percent (float, optional): Outlier percentage. Defaults to 0.5.
        p (float, optional): Probability that at least one random sample is free from outliers. Defaults to 0.98.
        t (int, optional): RANSAC threshold. Defaults to 10.

    Returns:
        numpy.ndarray: Estimated Homography of Model type. 3x3 matrix.
    """

    num_matches = CL1uv.shape[0]   # Number of matching points 

    if model == "Translation":
        dof = 2
    elif model == "Similarity":
        dof = 4
    elif model == "Affine":
        dof = 6
    elif model == "Projective":
        dof = 8        
    else:
        print("Invalid model")
        return None 
    
    if dof/2 > num_matches:
        print("Not enough matching points..")
        return None 
    
    best_inlier_idxs = None 
    best_consensus_percent = 0.0

    # Calculate the number of iterations according to the current number of estimated outliers and the target outlier percentage or the input num_iterations
    num_iterations = abs((log(1-p)) / (log(1-(1-outlier_percent)**(dof//2)) + 1e-6)) if num_iterations is None else num_iterations

    for i in range(int(num_iterations)):

        # Select a number of random point indices 
        points_indices = np.random.choice(num_matches, int(np.ceil(dof/2)), replace=False)  # Disables picking the same points twice

        # Estimate the Homography with the selected points
        CL1uv_random = CL1uv[points_indices]
        CL2uv_random = CL2uv[points_indices]
        H = compute_homography(CL1uv_random, CL2uv_random, model)

        if np.any(np.isnan(H)):
            #If there are problems with H estimation the consensus is valued as null
            consensus_percent = None

        else:
            # Compute the consesus related to estimated H
            errors = projection_error(H, CL2uv, CL1uv)
            inliers_indxs = np.where(errors < t)[0] # inliers
            consensus = np.sum(errors < t)
            consensus_percent = consensus / num_matches

            # Update best Homography found
            if consensus_percent > best_consensus_percent:
                best_inlier_idxs = inliers_indxs
                best_consensus_percent = consensus_percent

        # Exit condition
        if consensus_percent >= p: break

    # Estimate the Homography with the best inliers 
    INLIERS1uv = CL1uv[best_inlier_idxs].reshape(-1,2)
    INLIERS2uv = CL2uv[best_inlier_idxs].reshape(-1,2)
    H_best = compute_homography(INLIERS1uv, INLIERS2uv, model)
    
    return H_best, INLIERS1uv, INLIERS2uv
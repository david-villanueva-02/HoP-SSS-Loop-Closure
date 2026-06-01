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
        model (str): Type of Homography to estimate. It has to be equal to one of the following strings: 'Translation', 'Euclidean', 'Similarity', 'Affine', 'Projective'.
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
        min_sample = 1
    elif model == "Euclidean":
        dof = 3
        min_sample = 2
    elif model == "Similarity":
        dof = 4
        min_sample = 2
    elif model == "Affine":
        dof = 6
        min_sample = 3
    elif model == "Projective":
        dof = 8        
        min_sample = 4
    else:
        print("Invalid model")
        return None 
    
    if min_sample > num_matches:
        print("Not enough matching points..")
        return None 
    
    best_inlier_idxs = None 
    best_consensus_percent = 0.0

    # Calculate the number of iterations according to the current number of estimated outliers and the target outlier percentage or the input num_iterations
    if num_iterations is None:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        outlier_percent = min(max(outlier_percent, 1e-6), 1.0 - 1e-6)

        inlier_percent = 1.0 - outlier_percent
        good_sample_probability = inlier_percent ** min_sample

        denominator = log(1.0 - good_sample_probability)

        if abs(denominator) < 1e-12:
            num_iterations = 1
        else:
            num_iterations = int(np.ceil(log(1.0 - p) / denominator))

        num_iterations = max(num_iterations, 1)

    for i in range(int(num_iterations)):

        # Select a number of random point indices 
        points_indices = np.random.choice(num_matches, min_sample, replace=False)  # Disables picking the same points twice

        # Estimate the Homography with the selected points
        CL1uv_random = CL1uv[points_indices]
        CL2uv_random = CL2uv[points_indices]

        try:
            H = compute_homography(CL1uv_random, CL2uv_random, model)
        except Exception:
            H = None

        if H is None or np.any(np.isnan(H)) or np.any(np.isinf(H)):
            #If there are problems with H estimation the consensus is valued as null
            consensus_percent = None

        else:
            # Compute the consesus related to estimated H
            errors = projection_error(H, CL1uv, CL2uv)
            inliers_indxs = np.where(errors < t)[0] # inliers
            consensus = np.sum(errors < t)
            consensus_percent = consensus / num_matches

            # Update best Homography found
            if consensus_percent > best_consensus_percent:
                best_inlier_idxs = inliers_indxs
                best_consensus_percent = consensus_percent

            # Exit condition
            if consensus == num_matches: break

    if best_inlier_idxs is None or len(best_inlier_idxs) < min_sample:
        print("No valid consensus found..")
        return None, None, None

    # Estimate the Homography with the best inliers 
    INLIERS1uv = CL1uv[best_inlier_idxs].reshape(-1,2)
    INLIERS2uv = CL2uv[best_inlier_idxs].reshape(-1,2)
    H_best = compute_homography(INLIERS1uv, INLIERS2uv, model)
    
    return H_best, INLIERS1uv, INLIERS2uv
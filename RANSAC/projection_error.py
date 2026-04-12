import numpy as np

def projection_error(H12, CL1uv, CL2uv):
    """Given two list of coordinates (CL1uv and CL2uv) on two images and the
    homography that relates them (H12) this function will compute an error vector
    (errorVec).  This vector will contain the Euclidean distance between each
    point in CL1uv and its corresponding point in CL2uv after applying the
    homography  H12.

    Args:
        H12 (numpy.ndarray): Homography relating image #1 and image #2. 3x3 matrix.
        CL1uv (numpy.ndarray): Set of points on image #1. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.
        CL2uv (numpy.ndarray): Set of points on image #2. Each row represents a 2-D point (u,v). Size: Nx2, with N number of points.

    Returns:
        numpy.ndarray: Set of L2 norm's calculated between the original and projected points. Size: Nx1, with N number of points.
    """ 

    # Project coordinates of second image onto first one
    errors = []

    ### OLD ##
    # Calculate the error vector
    for i in range(len(CL1uv)):
        p1 = CL1uv[i]
        p2 = CL2uv[i]

        p2 = np.array([[p2[0]], [p2[1]], [1]]) # Homogenous
        projected_point = H12 @ p2

        # Calculate euclidean error
        du = p1[0] - projected_point[0,0]/projected_point[2,0]
        dv = p1[1] - projected_point[1,0]/projected_point[2,0]
        error = np.sqrt(du**2 + dv**2)
        errors.append(error)

    return np.array(errors)
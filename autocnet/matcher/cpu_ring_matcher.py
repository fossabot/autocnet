import numpy as np

def ransac_permute(ref_points, tar_points, tolerance_val, target_points):
    """
    Given a set of reference points and target points, compute the
    geometric distances between pairs of points in the reference set
    and pairs of points in the target set.  Points for which the ratio of
    the distances is within plus or minus 1 - tolerance_value are considered
    to be good matches.

    If a valid solution is not found, this func returns three empty lists.

    Parameters
    ----------
    ref_points : ndarray
                 A (n, 2) array of points where the first column is the x
                 pixel location and the second column is the y pixel location.
                 Additional columns are ignored. These points are from one
                 image.

    tar_points : ndarray
                 A (n,2) array as above, for a second image.

    tolerance_value : float
                      On the range [-1, 1], the computed ratio must be
                      within 1 +- tolerance to be considered a match

    target_points : int
                    The minimum number of points required to return
                    a valid answer

    Returns
    -------
    ref_points : ndarray
                 (n,2) subset of the input ref_points

    tar_points : ndarray
                 (n,2) subset of the input tar points

    f2 : ndarray
         of indices for valid points

    References
    ----------
    P. Sidiropoulos and J.-P. Muller, A systematic solution to multi-instrument co-registration of high-resolution planetary images to an orthorectified baseline, IEEE Transactions on Geoscience and Remote Sensing, 2017
    """
    n = len(ref_points)
    dist = np.zeros((n,n))
    for i in range(n):
        vr1 = ref_points[i]
        vt1 = tar_points[i]
        for j in range(i, n):
            # These are diagonals so always zero
            if i == j:
                dist[i,j] = 0
                dist[j,i] = 0
                continue
            # Compute the distance between ref_b - ref_a and tar_b - tar_a. The
            # absolute value should be small as these points should be sp
            vr2 = ref_points[j]
            vt2 = tar_points[j]

            dr = vr2 - vr1
            dt = vt2 - vt1

            dist[i,j] = (dr[0]**2 + dr[1]**2)**0.5 / (dt[0]**2+dt[1]**2)**0.5
            dist[j,i] = dist[i,j]
    minlim = 1 - tolerance_val
    maxlim = 1 + tolerance_val

    # Determine which points are within the tolerance
    q1 = dist > minlim
    q2 = dist < maxlim
    q = (q1*q2).astype(np.int)
    # How many points are within the tolerance?
    s = np.sum(q, axis=1)

    # If the number of points within the tolerance are greater than the number of desired pointsagreed re: scalars in the eqn. Hopefully ge
    if max(s) > target_points:
        m = np.eye(n).dot(target_points + 1)
        for i in range(n):
            for j in range(i):
                m[i,j] = q[i].dot(q[j])
                m[j,i] = m[i,j]
        qm = m > target_points
        sqm = np.sum(qm, axis=-1)
        f = np.argmax(sqm)
        f2 = np.nonzero(qm[f])
        return ref_points[f2], tar_points[f2], f2
    else:
        return [], [], []

def sift_match(a, b, thresh=1.5):
    """
    vl_ubcmatch from the vlfeat toolbox for MatLab.  This is
    Lowe's prescribed implementation for disambiguating descriptors.

    Parameters
    ----------
    a : np.ndarray
        (m,) a singular descriptors where the m-dimension are the
        descriptor lengths.  For SIFT m=128. This is reshaped from
        a vector to an array.
    b : np.ndarray
        (n,m) where the n-dimension are the individual features and
        the m-dimension are the elements of the descriptor.

    thresh : float
             The threshold for disambiguating correspondences. From Lowe.
             If best * thresh < second_best, a match has been found.

    Returns
    -------
    best : int
           Index for the best match

    References
    ----------
    P. Sidiropoulos and J.-P. Muller, A systematic solution to multi-instrument co-registration of high-resolution planetary images to an orthorectified baseline, IEEE Transactions on Geoscience and Remote Sensing, 2017
    """
    a = a.reshape(1,-1)
    dists = np.sum((b-a)**2, axis=1)

    try:
        best = np.nanargmin(dists)
    except:
        return

    if len(dists[dists != dists[best]]) == 0:
        return  # Edge case where all descriptors are the same
    elif len(dists[dists == dists[best]]) > 1:
        return  # Edge case where the best is ambiguous
    sec_best = np.nanmin(dists[dists != dists[best]])

    if dists[best] * thresh < sec_best:
        return best
    return

def ring_match(ref_feats, tar_feats, ref_desc, tar_desc, ring_radius=4000, max_radius=40000, target_points=15, tolerance_val=0.02):
    """
    Apply the University College London ring matching technique that seeks to match
    target feats to a number of reference features.

    Parameters
    ----------
    ref_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the target features

    tar_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the reference features

    ref_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    tar_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    ring_radius : numeric
                  The width of a ring for matching. In the same units as the x,y
                  coordinates for the features, e.g. if the ref_feats and tar_feats
                  are provided in pixel space and meters, the ring_radius should
                  be expressed in meters

    max_radius : numeric
                 The maximum radius to be tested.  This is the maxixum distance
                 a given correspondence could be from the initial estimate.

    target_points : int
                    The number of points that constitute a valid match

    tolerance : float
                The tolerance for outlier detection in point ordering between estimated
                resolutions

    Returns
    -------
    xref : ndarray
           (n,4) array of the correspondences selected from the ref_feats input

    xtar : ndarray
           (n,4) array of the correspondences selected from the tar_feats input

    p_idx : ndarray
            (n,2) array of the indices in the ref_feats and tar_feats input arrays

    References
    ----------
    P. Sidiropoulos and J.-P. Muller, A systematic solution to multi-instrument co-registration of high-resolution planetary images to an orthorectified baseline, IEEE Transactions on Geoscience and Remote Sensing, 2017
    """

    # Reference and target features
    ref_xy = ref_feats[:,:2]
    ref_xmym = ref_feats[:,3:]
    tar_xy = tar_feats[:,:2]
    tar_xmym = tar_feats[:,3:]

    # Boolean mask for those reference points that have already been matched
    ref_mask = np.ones(len(ref_xy), dtype=bool)

    # Counters
    numr = len(ref_feats)

    # Number of radial rings
    rad_num = int(max_radius / ring_radius)

    # Number of points per ring vector
    points_num = np.zeros(rad_num, dtype=np.int)

    # Initial array for holding candidate points - this is grown dynamically below
    p = np.zeros((target_points, 4 * rad_num))
    p_idx = np.zeros((target_points, 2 * rad_num), dtype=np.int)

    # Increment counter for determining how frequently to assess rings
    metr = 1
    # Main processing
    while ref_mask.any():
        # Grab a random reference point
        r = np.random.choice(np.arange(numr)[ref_mask])
        current_ref_desc = ref_desc[r]
        current_ref_xy = ref_xy[r]
        current_ref_xmym = ref_xmym[r]

        # Compute the euclidean distance between the reference point and all targets
        d = np.linalg.norm(current_ref_xmym - tar_xmym, axis=1)

        # For each point, independently match to a point in a given ring
        for i in range(rad_num):
            # The number of points that are within a given ring
            z = (d > i * ring_radius) * (d < (i+1) * ring_radius)
            # If we have enough points, run the sift matcher and select the best point, updating p
            if np.sum(z) > target_points:
                # All candidate points that are in the ring
                current_tar_descs = tar_desc[z]  # This slicing uses ~25% of processing timr
                current_tar_xys = tar_xy[z]
                z_idx = np.where(z == True)[0]
                #assert sum(z) == current_tar_descs.shape[0] == current_tar_xys.shape[0]

                # Sift Match
                match = sift_match(current_ref_desc, current_tar_descs, thresh=1.5)  # The remaining 75% of processing time.

                if match is not None:
                    if points_num[i] == p.shape[0]:
                        # Inefficient, but creates a dynamically allocated array the larger array_step is, the less inefficient this should be
                        p_append = np.zeros((target_points, 4*rad_num))
                        p = np.vstack((p, p_append))
                        p_idx_append = np.zeros((target_points,2*rad_num), dtype=np.int)
                        p_idx = np.vstack((p_idx, p_idx_append))
                    p[points_num[i], 4*i:4*i+4] = [current_ref_xy[0], current_ref_xy[1], current_tar_xys[match][0], current_tar_xys[match][1]]
                    # Set the id of the point
                    p_idx[points_num[i], 2*i:2*i+2] = [r, z_idx[match]]
                    points_num[i] += 1

        #For every 200 reference points that are potentially matched
        if metr % 200 == 0:
            max_cons = 3
            # Find all candidate rings
            candidate_rings = points_num >= target_points
            for j in range(rad_num):
                if candidate_rings[j]:
                    # For each ring that is a candidate, select all of the reference and targets from the p matrix
                    # the first part of the slice grabs all candidate points and the second half in the first two args
                    # selects the ref and target (respectively).
                    npoints_in_ring = points_num[j]
                    ref_points = p[:npoints_in_ring, 4*j:4*j+2] # Slice out the reference coords
                    tar_points = p[:npoints_in_ring, 4*j+2:4*j+4]  # Slice out the target coords
                    indices = p_idx[:npoints_in_ring, 2*j:2*j+2] # slice out the indices
                    xref, xtar, idx = ransac_permute(ref_points, tar_points, tolerance_val, target_points)
                    # This selects the best of the rings
                    max_cons = max(max_cons, len(xref))
                    if len(xref) >= target_points:
                        # Solution found
                        # Instead of returning the ref and tar points, return the ids
                        ring = (j*ring_radius, j*ring_radius+ring_radius)
                        return xref, xtar, indices[idx], ring
        # Mask the reference point and iterate to the next randomly selected point
        ref_mask[r] = False
        metr += 1
    return None, None, None

def directed_ring_match(ref_feats, tar_feats, ref_desc, tar_desc, ring_min, ring_max, target_points=15, tolerance_value=0.02):
    """
    Given an input set of reference features and target features, attempt to find
    correspondences within a given ring, where the ring is defined by min and max
    radii.  This is a directed version of the ring_match function, in that this
    function assumes that the correspondence is within the defined ring.

    This implementation is inspired by and uses the ring_match implementation above, developed
    by Sidiropoulos and Muller.

    Parameters
    ----------
    ref_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the target features

    tar_feats : np.ndarray
                where the first 2 columns are the x,y coordinates in pixel space,
                columns 3 & 4 are the x,y coordinates in m or km in the same
                reference as the reference features

    ref_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    tar_desc : np.ndarray
               (m, n) where m are the individual SIFT features and
               n are the descriptor length (usually 128)

    ring_min : numeric
               The inner distance of the ring

    ring_max : numeric
               The outer distance of the ring


    target_points : int
                    The number of points that constitute a valid match

    tolerance : float
                The tolerance for outlier detection in point ordering between estimated
                resolutions

    Returns
    -------
    xref : ndarray
           (n,4) array of the correspondences selected from the ref_feats input

    xtar : ndarray
           (n,4) array of the correspondences selected from the tar_feats input

    p_idx : ndarray
            (n,2) array of the indices in the ref_feats and tar_feats input arrays
    """

    # Reference and target features
    ref_xy = ref_feats[:,:2]
    ref_xmym = ref_feats[:,3:]
    tar_xy = tar_feats[:,:2]
    tar_xmym = tar_feats[:,3:]

    numr = len(ref_feats)

    p = np.zeros((numr, 4))
    p_idx = np.zeros((numr, 2), dtype=np.int)
    points_num = 0
    # Iterate over all of the reference points seeking a match
    for r in range(numr):
        current_ref_desc = ref_desc[r]
        current_ref_xy = ref_xy[r]
        current_ref_xmym = ref_xmym[r]

        # Compute the euclidean distance between the reference point and all targets
        d = np.linalg.norm(current_ref_xmym - tar_xmym, axis=1)
        # Find all of the candidates, if none, skip
        z = (d > ring_min) * (d < ring_max)

        if np.sum(z) == 0:
            continue
        current_tar_descs = tar_desc[z]
        current_tar_xys = tar_xy[z]
        z_idx = np.where(z == True)[0]

        match = sift_match(current_ref_desc, current_tar_descs, thresh=1.5)
        if match is not None:
            p[points_num] = [current_ref_xy[0], current_ref_xy[1], current_tar_xys[match][0], current_tar_xys[match][1]]
            # Set the id of the point
            p_idx[points_num] = [r, z_idx[match]]
            points_num += 1
    if points_num == 0:
        # No candidate matches found in this set.
        return [], [], []
    # Now that the candidates have all been located, check their geometric relationships to find the good matches
    ref_points = p[:points_num, :2]
    tar_points = p[:points_num, 2:]
    xref, xtar, idx = ransac_permute(ref_points, tar_points, tolerance_value, target_points)
    return xref, xtar, p_idx[idx]

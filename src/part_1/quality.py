def homography_quality(matches, inliers):
    if matches == 0: # no match, no calculation
        ratio = 0.0  # set to zero
    else:
        ratio = inliers / matches # accepted matches by RANSAC

    if inliers >= 100 and ratio >= 0.30:  # strong result
        quality = "good"                  # save result
    elif inliers >= 50 and ratio >= 0.20: # fewer but acceptable
        quality = "medium"                # save result
    else:
        quality = "bad"                   # skip or stop

    return {
        "matches": matches, # total feature matches before ransac filtering
        "inliers": inliers, # number of matches after ransac filtering
        "ratio": ratio,     # ratio num of real matches
        "quality": quality  # good, medium or bad
    }
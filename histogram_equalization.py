import gc
import numpy as np
import matplotlib.pyplot as plt

# Code in this file is based on or written by 
# Gabriel Eilertsen

import numpy as np
import gc

def histeq(x, bins=256, batch_size=100):
    """
    Perform histogram equalization on a set of images.

    Parameters:
    x : list of ndarray
        List of images to be equalized.
    bins : int
        Number of bins for the histogram.
    batch_size : int
        Number of images to process in each batch.

    Returns:
    t : ndarray
        Histogram equalization function.
    bc : ndarray
        Bin centers.
    """
    
    print("Computing histogram equalization function...")
    # Bin edges (for histogram computation)
    b = np.linspace(0, 1, bins + 1)

    # Bin centers (for interpolation)
    bc = (b[1:] + b[:-1]) / 2
    
    # Number of images
    N = len(x)

    # Initialize histogram
    H = np.zeros(bins)

    # Process images in batches for normalization and histogram calculation
    num_batches = (N + batch_size - 1) // batch_size  # Calculate number of batches
    print("Processing images in " + str(num_batches) + " batches...")
    for batch_idx in range(num_batches):
        print("Batch " + str(batch_idx + 1) + "/" + str(num_batches))
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, N)
        for i in range(start_idx, end_idx):
            # Normalize the image
            image = x[i]
            normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            # Calculate histogram for the normalized image and accumulate
            h, _ = np.histogram(normalized_image, bins=b)
            H += h
            
            # Free memory
            del normalized_image
            del h
            gc.collect()

    # Normalize the histogram
    H = H / np.sum(H)

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(H)

    # Interpolate the CDF to get the equalization function
    t = np.interp(bc, bc, cdf)

    return t, bc

# Example usage

#t, bc = histeq(x, bins=100)

# # Apply histogram equalization
# xt = np.interp(x, bc, t)

# # Invert histogram equalization
# xti = np.interp(xt, t, bc)
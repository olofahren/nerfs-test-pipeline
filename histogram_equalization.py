import gc
import numpy as np
import matplotlib.pyplot as plt

# Code in this file is based on or written by 
# Gabriel Eilertsen

def histeq(x, bins=50, batch_size=100):
    """Compute histogram equalization function.
    
    Parameters
    ----------
    x : ndarray
        Input data. A 4D array of all images in the scene/dataset
    bins : int
        Number of bins in the histogram.
        
    Returns
    -------
    t : ndarray
        Histogram equalization function.
    bc : ndarray
        Bin centers.
    """
    
    print("Computing histogram equalization function...")
    # Bin edges (for histogram computation)
    b = np.linspace(0,1,bins+1)

    # Bin centers (for interpolation)
    bc = (b[1:]+b[:-1])/2
    
    # Number of images
    N = len(x)

    #NOTE: Memory intensive operation, perhaps fix this. Works for now, but might be a problem for larger datasets
    print("Normalizing images...")
    for i in range(N):
        x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
        
    

    # Initialize histogram
    H = np.zeros(bins)

    # Process images in batches
    num_batches = (N + batch_size - 1) // batch_size  # Calculate number of batches
    print("Calculating histogram for images in "+ str(num_batches)+" batches...")
    for batch_idx in range(num_batches):
        print("Batch "+str(batch_idx+1)+"/"+str(num_batches))
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, N)
        batch = x[start_idx:end_idx]

        # Calculate histogram for each image in the batch and accumulate
        for image in batch:
            h, _ = np.histogram(image, bins=b)
            H += h
            del h  # Free memory
            gc.collect()
    
        del batch
        gc.collect()
    
    # H = H / N
    
    # Histogram equalization function (CDF)
    t = np.cumsum(H)/np.sum(H)
    # t = t / t[-1]

    print("Histogram equalization function computed!")

    # Plotting for debugging
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.bar(bc, H, width=0.8 * (bc[-1] - bc[0]) / bins)
    plt.title('Average histogram')
    plt.subplot(1, 2, 2)
    plt.plot(bc, t)
    plt.plot(t, bc)
    plt.title('Histogram equalization function')
    plt.legend(['inverse', 'forward'])
    plt.show()
    
    return t, bc

# Example usage

#t, bc = histeq(x, bins=100)

# # Apply histogram equalization
# xt = np.interp(x, bc, t)

# # Invert histogram equalization
# xti = np.interp(xt, t, bc)
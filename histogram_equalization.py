import numpy as np
import matplotlib.pyplot as plt

# Code in this file is based on or written by 
# Gabriel Eilertsen

def histeq(x, bins=50):
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
    N = x.shape[0]

    # Histogram
    # If the set of images are in 4-D array, the histogram can be computed
    # directly over the whole set
    H, _ = np.histogram(x, bins=b)
    
    # Otherwise, individual histograms can be averaged:
    # H = np.zeros(bins)
    # for i in range(N):
    #    h, _ = np.histogram(x[i], bins=b)
    #    H += h
    
    # Histogram equalization function (CDF)
    t = np.cumsum(H)/np.sum(H)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.bar(bc,H,width=0.8*(bc[-1]-bc[0])/bins)
    plt.title('Average histogram')
    plt.subplot(1,2,2)
    plt.plot(bc,t)
    plt.plot(t,bc)
    plt.title('Histogram equalization function')
    plt.legend(['inverse','forward'])
    plt.show()
    
    return t, bc

# Example usage

#t, bc = histeq(x, bins=100)

# # Apply histogram equalization
# xt = np.interp(x, bc, t)

# # Invert histogram equalization
# xti = np.interp(xt, t, bc)
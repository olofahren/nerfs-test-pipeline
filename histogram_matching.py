import os
import time
from matplotlib import pyplot as plt
import numpy as np
import gc
from PIL import Image
#from dataaug_test import loadImagesWithFilenames


def calculateAverageHistogram(images, bins=256, batch_size=100):
    
    H = np.zeros(bins)
    print("Calculating average histogram...")
    for image in images:
        #progress bar that shows the progress of the histogram calculation
        print("Calculating histogram for image " + str(images.index(image) + 1) + "/" + str(len(images)), end='\r')
        
        image = np.array(image[1])
        
        #progeress bar that shows the progress of the histogram calculation
        # Normalize the image
        #normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
                
        # Calculate histogram for the normalized image and accumulate
        h, _ = np.histogram(image, bins)
        H += h
        
        # Free memory
        #del normalized_image
        del h
        gc.collect()

    # Normalize the histogram between 0 and 1
    H = (H / np.sum(H))
    return H



def histMatch(images, std = 0.1, bins=256, batch_size=100):
    startTime = time.time()
    histMatch_images = []
    #calculate the average histogram of the images
    h = calculateAverageHistogram(images, bins, batch_size)
    
    b = np.linspace(0, 1, bins + 1)

    # Bin centers (for interpolation)
    bc = (b[1:] + b[:-1]) / 2

    h_target = np.exp(-0.5*((bc-0.5)/std)**2) # Gaussian histogram
    #scale the target histogram to the same size as the input histogram
    h_target = h_target/np.sum(h_target)*np.sum(h)
    
    t = np.cumsum(h)/np.sum(h) # For histogram equalization
    t_target = np.cumsum(h_target)/np.sum(h_target) # For matching to Gaussian histogram
    
    #plot the average histogram and the target histogram
    
    
    
    #plt.plot(bc, h, label='Average histogram')
    #plt.plot(bc, h_target, label='Target histogram')
    #plt.legend()
    #plt.show()
    
    print("Histogram matching...")
    for image in images:
        print("Matching histogram for image " + str(images.index(image) + 1) + "/" + str(len(images)), end='\r')
        image_data = np.array(image[1])/255
        #normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        xt = np.interp(image_data, bc, t)  # Mapping from input to uniform distribution (histogram equalization)
        xt = np.interp(xt, t_target, bc)  # Mapping from uniform distribution to Gaussian
        
        #xt = (xt - np.min(xt)) / (np.max(xt) - np.min(xt))
        xt = Image.fromarray((xt * 255).astype(np.uint8))
        
        #replace the original image with the transformed image
        images[images.index(image)] = (image[0], xt)
        
        
        del image_data
        #del normalized_image
        del xt
        gc.collect()
    
    print("Histogram matching done")
    print("Time taken: ", time.time() - startTime)
    time.sleep(5)
    
    #calculate the new average histogram
    newAvgHist = calculateAverageHistogram(images, bins, batch_size)
    
    #plot the the average histogram, the target histogram and the transformed histogram
    # plt.plot(bc, h, label='Average histogram')
    # plt.plot(bc, h_target, label='Target histogram')
    # plt.plot(bc, newAvgHist, label='Transformed histogram')
    # plt.legend()
    # plt.show()
    
    return images, bc, t_target, t, newAvgHist
    
    
    
    
    
# #Test usage
# root = "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/riverview/images-jpeg-1k/"
# root = os.path.expanduser(root)
# training_images = loadImagesWithFilenames(root, "eyefulTower")

# #calculate the average histogram of the images
# orgAverageHistogram = calculateAverageHistogram(training_images)

# compareImage = training_images[0][1]
# inverseHistogramMatchedImages = []

# images, bc, t_target, t, newAvgHist = histMatch(training_images, std=0.4)

# for image in images:
#         print("Inverse histogram matching image " + str(images.index(image) + 1) + "/" + str(len(images)), end='\r')
#         # Extract the image data
#         image_data = np.array(image[1])/255
        
#         # Apply inverse histogram matching
#         xti = np.interp(image_data, bc, t_target)  # Mapping from Gaussian distribution to uniform
#         corrected_image_data = np.interp(xti, t, bc)  # Mapping from uniform distribution to original
        
#         # Scale back to [0, 255]
#         corrected_image_data = (corrected_image_data * 255).astype(np.uint8)
#         corrected_image = Image.fromarray(corrected_image_data)
#         inverseHistogramMatchedImages.append((image[0], corrected_image))
        
# inverseHistogram = calculateAverageHistogram(inverseHistogramMatchedImages)
        
        
# #display the original histgoram and the new average histogram
# plt.plot(bc, orgAverageHistogram, label='Original average histogram')
# plt.plot(bc, newAvgHist, label='New average histogram')
# plt.plot(bc, inverseHistogram, label='Average histogram of inverse transformed images')
# plt.legend()
# plt.show()

# #show the first image in training_images, the transformed image and the inverse transformed image nex to each other
# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(compareImage)
# axs[0].set_title("Original image")
# axs[1].imshow(images[0][1])
# axs[1].set_title("Transformed image")
# axs[2].imshow(inverseHistogramMatchedImages[0][1])
# axs[2].set_title("Inverse transformed image")
# plt.show()




    
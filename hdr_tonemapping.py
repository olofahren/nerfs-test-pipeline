import gc
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

import gc
import os
import cv2
import numpy as np
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt

def generate_normal_distribution_image(shape, mean=0.5, std=0.1):
    """
    Generate a normally distributed image with the given shape, mean, and standard deviation.
    """
    normal_image = np.random.normal(mean, std, shape)
    normal_image = np.clip(normal_image, 0, 1)  # Clip values to the range [0, 1]
    return normal_image

def quantize_image(image, levels):
    """
    Quantize the image to the specified number of levels.
    """
    quantized_image = np.floor(image * (levels - 1)) / (levels - 1)
    return quantized_image

def hdrTonemap2Histogram(images, levels=256):
    """
    images : list of ndarray
        The input images to be processed.
    levels : int
        The number of quantization levels.
    """
    for idx, image in enumerate(images):
        print(f"Matching histogram for image {idx + 1}/{len(images)}", end='\r')
        
        try:
            # Normalize the HDR image to the range [0, 1]
            normalized_image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
            
            # Generate a normally distributed reference image
            ref_image = generate_normal_distribution_image(normalized_image.shape)
            
            # Check if the image is multi-channel
            if len(normalized_image.shape) == 3 and normalized_image.shape[2] == 3:
                print(f"Image {idx + 1} is multi-channel")
                # Match the histogram for the multi-channel image
                matched_image = match_histograms(normalized_image, ref_image)
            else:
                print(f"Image {idx + 1} is single-channel")
                # Match the histogram for the single-channel image
                matched_image = match_histograms(normalized_image, ref_image)
            
            # Quantize the histogram-matched image
            quantized_image = quantize_image(matched_image, levels)
            
            # Overwrite the original image in the list
            images[idx] = quantized_image
        except Exception as e:
            print(f"Error processing image {idx + 1}: {e}")
            continue

    return images


#load exr images using cv2.imread with IMREAD_ANYDEPTH flag
def loadImagesWithFilenames(folder):    
    """
    folder : str
        The folder containing the images.
    prefix : str
        The prefix of the images to load.
    """
    folder = os.path.expanduser(folder)

    
    images = []
    counter = 0
    for root, dirs, files in os.walk(folder):
        #do not load the images in original_images folder
        
        if "original_images" in root:
            continue
        for filename in files:
            counter += 1
            #print("Include list provided")
            if counter > 100:
                break
            if filename.endswith(('.exr')):
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append((filename, img))
                except IOError:
                    print("Error opening image " + filename)
                        
    return images


def calculateAverageHistogram(images, bins=256, batch_size=100):
    
    H = np.zeros(bins)
    print("Calculating average histogram...")
    for image in images:
        #progress bar that shows the progress of the histogram calculation
        #print(f"Calculating histogram for image {idx + 1}/{len(images)}", end='\r')
        
        image = np.array(image)
   
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

# Example usage
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

folder = "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/riverview/images-1k/"

nonToneMappedImages = loadImagesWithFilenames(folder)

# Extract only the images
nonToneMappedImages = [image[1] for image in nonToneMappedImages]

bins = 256

toneMappedImages = hdrTonemap2Histogram(nonToneMappedImages, levels=256)

# Calculate the average histogram of the tone-mapped images
avgHistogramMatchedImages = calculateAverageHistogram(toneMappedImages, bins, 100)

# Print the shape of the first tone-mapped image
print(toneMappedImages[0].shape)

# Show 5 random images in a grid
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
    axs[i].imshow(toneMappedImages[i])
    axs[i].axis('off')
plt.show()

# Generate a normally distributed reference histogram for comparison
ref_image = generate_normal_distribution_image(toneMappedImages[0].shape)
ref_histogram, _ = np.histogram(ref_image.ravel(), bins, density=True)

plt.plot(ref_histogram, label='Target histogram')
plt.plot(avgHistogramMatchedImages, label='Transformed histogram')
plt.legend()
plt.show()
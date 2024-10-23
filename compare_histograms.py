from matplotlib import pyplot as plt
import numpy as np
from histogram_matching import histMatch
from histogram_matching import calculateAverageHistogram
from dataaug_test import loadImagesWithFilenames

# Load images
images = loadImagesWithFilenames("/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-27_143710/test/rgb/", "other")
#normalize the images
#images = [(image[0], (image[1] - np.min(image[1])) / (np.max(image[1]) - np.min(image[1])) ) for image in images]
bins = 256

hist = calculateAverageHistogram(images, bins, 100)

print(hist)

std = 0.1
b = np.linspace(0, 1, bins + 1)
# Bin centers (for interpolation)
bc = (b[1:] + b[:-1]) / 2
h_target = np.exp(-0.5*((bc-0.5)/std)**2) # Gaussian histogram

plt.plot(hist, label='Average histogram')
plt.plot(h_target, label='Gaussian histogram')
plt.legend()
plt.show()




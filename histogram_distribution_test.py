import sys

from matplotlib import pyplot as plt
import numpy as np
from dataaug_test import loadImagesWithFilenames
from PIL import Image
import os
from gamma_correction import adjustGamma

def calculateHistogram(images, noOfBins=256):
    H = np.zeros(noOfBins)

    #caluclate histograms for all images and average them  
    
    for image in images:
        sys.stdout.write("\rCalculating histograms: " + str(images.index(image)) + "/" + str(len(images)-1))
        sys.stdout.flush()
        image = np.array(image[1])              
        #normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        h, _ = np.histogram(image, noOfBins)
        H += h
    sys.stdout.write("\n")
        
    H = H / np.sum(H)
    return H    

def displayHistogram(H,H2, title1="Histogram 1", title2="Histogram 2", gammaValue=None, sceneName=None):
    #plot the histograms in the same figure
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle("Histogram Comparison")
    axs[0].bar(np.linspace(0,255,256), H)
    axs[0].set_title(title1)
    
    axs[1].bar(np.linspace(0,255,256), H2)
    axs[1].set_title(title2)
    
    if gammaValue is not None:
        fig.text(0.5, 0.01, "Gamma value: " + str(gammaValue), ha='center')
    if sceneName is not None:
        fig.text(0.5, 0.05, "Scene: " + sceneName, ha='center')
    plt.show()
    

def estimateOptimalGammaValue(dataFolder):
    dataFolder = os.path.expanduser(dataFolder)
    images = loadImagesWithFilenames(dataFolder,"eyefulTower")
    avgPixelIntensity = 0
    
    for image in images:
        image = np.array(image[1])
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        avgPixelIntensity += np.mean(normalized_image)
        
    avgPixelIntensity = avgPixelIntensity / len(images)
    
    #calculate gamma value
    gamma = np.log(0.5) / np.log(avgPixelIntensity)
    print("Estimated best gamma value: ", gamma)    
    return gamma



filePath = "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/raf_furnishedroom/images-jpeg-1k"
sceneName = "raf_furnishedroom"
    
originalHist = calculateHistogram(loadImagesWithFilenames(filePath, "eyefulTower"))

estimatedGamma = estimateOptimalGammaValue(filePath)

gammaCorrectedImages = adjustGamma(estimatedGamma, loadImagesWithFilenames(filePath,"eyefulTower"))

gammaCorrectedHist = calculateHistogram(gammaCorrectedImages)

#plot the histograms next to each other

displayHistogram(originalHist, gammaCorrectedHist, "Original Histogram", "Gamma Corrected Histogram", estimatedGamma, sceneName)
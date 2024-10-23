from PIL import Image
import os
import numpy as np
from dataaug_test import loadImagesWithFilenames

def calculateErrorImages(groundTruthImages, renderedImages):
    errorImages = []
    for i in range(len(groundTruthImages)):
        #progress bar
        print("Calculating error images: " + str(i+1) + "/" + str(len(groundTruthImages)), end="\r")
        groundTruthImage = groundTruthImages[i][1]
        renderedImage = renderedImages[i][1]
        errorImage = Image.new('RGB', groundTruthImage.size)
        for x in range(groundTruthImage.size[0]):
            for y in range(groundTruthImage.size[1]):
                groundTruthPixel = groundTruthImage.getpixel((x, y))
                renderedPixel = renderedImage.getpixel((x, y))
                errorPixel = (abs(groundTruthPixel[0] - renderedPixel[0]), abs(groundTruthPixel[1] - renderedPixel[1]), abs(groundTruthPixel[2] - renderedPixel[2]))
                errorImage.putpixel((x, y), errorPixel)
        errorImages.append((groundTruthImages[i][0], errorImage))
    return errorImages





renderedImagesFilepath = "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-20_051051/test/rgb/"
groundTruthImagesFilepath = "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/office_view1/images-jpeg-1k/22"




groundTruthImages = loadImagesWithFilenames(groundTruthImagesFilepath, "none")
renderedImages = loadImagesWithFilenames(renderedImagesFilepath, "none")

#groundTruthImages[0][1].show()
#renderedImages[0][1].show()


errorImages = calculateErrorImages(groundTruthImages, renderedImages)


#show all error images in a clickable window
for errorImage in errorImages:
    #errorImage[1].show()
    #create a folder for the error images
    if not os.path.exists(renderedImagesFilepath+"/error_images"):
        os.makedirs(renderedImagesFilepath+"/error_images")
    errorImage[1].save(renderedImagesFilepath+"/error_images/" + errorImage[0])
    print("Error image saved as " + errorImage[0])

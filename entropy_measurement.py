import os
import dataaug_test
import gamma_correction
from skimage.measure import shannon_entropy

def calculateAvgSceneEntropy(images):
    """
    Calculate the average entropy of a scene given a list of images.

    Parameters:
    images (list): A list of tuples where each tuple contains the filename and the image data.

    Returns:
    float: The average entropy of the scene.
    """
    scene_entropy = 0
    for image in images:
        scene_entropy += shannon_entropy(image[1])
    
    scene_entropy = scene_entropy / len(images)
    
    return scene_entropy

gammaValues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

#"/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/office1b/images-jpeg-1k/",
#"/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/riverview/images-jpeg-1k/",
#"/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/office_view2/images-jpeg-1k/",
folderPaths = [
    #"/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/raf_furnishedroom/images-jpeg-1k/",
    "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/apartment/images-jpeg-1k/",
    "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/office_view1/images-jpeg-1k/"
]

batch_size = 10  # Define the batch size

for folderPath in folderPaths:
    folderPath = os.path.expanduser(folderPath)
    # Loading images
    images = dataaug_test.loadImagesWithFilenames(folderPath, "eyefulTower", [])
    print("Loaded ", len(images), " images")

    # Run gamma correction on the image dataset
    print("Calculating entropy for images...")
    entropy = []  # Initialize entropy list for each folder path
    for gamma in gammaValues:
        print("Performing gamma correction with gamma value: ", gamma)
        batch_entropy = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            gammaCorrectedImages = gamma_correction.adjustGamma(gamma, batch)
            
            print(f"Calculating average scene entropy for gamma value: {gamma}, batch {i // batch_size + 1}")
            avg_entropy = calculateAvgSceneEntropy(gammaCorrectedImages)
            batch_entropy.append(avg_entropy)
            print("Batch Entropy: ", avg_entropy)

            # Clear gammaCorrectedImages to free up memory
            del gammaCorrectedImages

        # Calculate the overall average entropy for the current gamma value
        overall_avg_entropy = sum(batch_entropy) / len(batch_entropy)
        entropy.append(overall_avg_entropy)
        print("Overall Entropy: ", overall_avg_entropy)

    for i in range(len(gammaValues)):
        print("Gamma: ", gammaValues[i], " Entropy: ", entropy[i])

    # Write the results to a file in append mode
    with open("entropy_results.txt", "a") as file:
        # Write the scene file path
        file.write(folderPath + "\n")
        for i in range(len(gammaValues)):
            file.write(str(entropy[i]) + "\n")
        file.write("\n")

    # Clear entropy list to free up memory
    del entropy

    # Clear images to free up memory
    del images
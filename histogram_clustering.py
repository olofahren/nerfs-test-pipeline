import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from dataaug_test import loadImagesWithFilenames

def clusterImagesBasedOnHistograms(scenePath, numClusters=3, batch_size=100, plot=False):
    
    Nbin = 50  # Number of histogram bins
    Nc = numClusters  # Number of clusters
    
    scenePath = os.path.expanduser(scenePath)
    images = loadImagesWithFilenames(scenePath, "eyefulTower", [])
    print("Loaded %d images from %s" % (len(images), scenePath))
    
    b = np.linspace(0, 1, Nbin + 1)  # bin edges
    bc = (b[1:] + b[:-1]) / 2  # bin centers

    # Initialize an empty list to store histograms
    histograms = []

    # Process images in batches
    print("Calculating histograms for images in batches...")
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        for img in batch_images:
            image_array = np.asarray(img[1]) / 255.0  # Normalize the image
            image_array = image_array.flatten()  # Flatten the image
            h, _ = np.histogram(image_array, bins=b)
            histograms.append(h)
    
    # Convert the list of histograms to a NumPy array
    C = np.array(histograms)
    
    # Cluster histograms into Nc different clusters
    print("Clustering histograms...")
    kmeans = KMeans(n_clusters=Nc, random_state=0).fit(C)

    # Labels of each histogram, i.e. which cluster each histogram belongs to (cluster 0,1,...,Nc-1)
    kl = kmeans.labels_
    
    #combine the cluster labels with the images and the filenames
    imagesWithFileNamesAndClusterLabel = [None] * len(images)
    for i in range(len(images)):
        imagesWithFileNamesAndClusterLabel[i] = (images[i][0], images[i][1], kl[i])

    # Plot the clustered images
    if plot:
        m = np.zeros(Nc)
        for c in range(Nc):
            # Distance between this cluster center and the avg. histograms
            m[c] = np.mean(np.power(kmeans.cluster_centers_[c,:]-np.mean(C), 2))
            
            plt.subplot(1,Nc,c+1)
            plt.title('Cluster %d, %d histograms, d=%0.4f'%(c,np.sum(kl==c),m[c]))
            
            # Plot individual histograms of this cluster
            for i in range(len(kl)):
                if kl[i] == c:
                    plt.plot(bc,C[i,:],color=[0.3,0.3,0.3],alpha=0.1)
            
            # Plot the cluster center histogram
            plt.plot(bc,kmeans.cluster_centers_[c,:],color=[0.8,0.3,0.3],linewidth=3)
            
            plt.grid(1)
            plt.ylim([0,1.5*np.max(kmeans.cluster_centers_[c,:])])
        plt.show()
        
        # plot the images in each cluster in a grid. One window for each cluster
        for c in range(Nc):
            plt.figure()
            plt.suptitle('Cluster %d, %d histograms, d=%0.4f'%(c,np.sum(kl==c),m[c]))
            count = 0
            for i in range(len(kl)):
                if kl[i] == c:
                    plt.subplot(10,10,count+1)
                    plt.imshow(images[i][1])
                    plt.axis('off')
                    count += 1
                    if count >= 100:
                        break
            plt.show()

    return imagesWithFileNamesAndClusterLabel


def calculateAverageQualityMeasuresGivenCluster(csvQualityFile, scenePath, numClusters):
    
    clusteredImages = clusterImagesBasedOnHistograms(scenePath, numClusters=3, batch_size=100, plot=True)
    
    file = open(csvQualityFile, "r")
    lines = file.readlines()
    file.close()
    
    avgPSNR0, avgSSIM0, avgLPIPS0 = 0, 0, 0
    avgPSNR1, avgSSIM1, avgLPIPS1 = 0, 0, 0
    avgPSNR2, avgSSIM2, avgLPIPS2 = 0, 0, 0
    counter0, counter1, counter2 = 0, 0, 0
    
    #if the filename is both in the quality file and in the clusteredImages list, add the quality measures to the average
    for line in lines:
        filename = line.split(",")[0]
        for img in clusteredImages:
            if img[0] == filename and img[2] == 0:
                avgPSNR0 += float(line.split(",")[1])
                avgSSIM0 += float(line.split(",")[2])
                avgLPIPS0 += float(line.split(",")[3])
                counter0 += 1
            elif img[0] == filename and img[2] == 1:
                avgPSNR1 += float(line.split(",")[1])
                avgSSIM1 += float(line.split(",")[2])
                avgLPIPS1 += float(line.split(",")[3])
                counter1 += 1
            elif img[0] == filename and img[2] == 2:
                avgPSNR2 += float(line.split(",")[1])
                avgSSIM2 += float(line.split(",")[2])
                avgLPIPS2 += float(line.split(",")[3])
                counter2 += 1
                
    return [avgPSNR0/counter0, avgSSIM0/counter0, avgLPIPS0/counter0], [avgPSNR1/counter1, avgSSIM1/counter1, avgLPIPS1/counter1], [avgPSNR2/counter2, avgSSIM2/counter2, avgLPIPS2/counter2]


# Example usage
scenePath = "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/riverview/images-jpeg-1k/"

# csvFilePaths = ["/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_092634/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_095223/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_101735/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_104248/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_124236/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_130838/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_133501/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_140821/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_143524/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_150132/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_153741/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_160348/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_162900/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_165412/test/rgb/",
#                 "/home/exjobb/oloah408/nerfs-test-pipeline/renders/2024-09-09_171921/test/rgb/",]

#get the latest folder from the renders directory
rendersPath = "/home/exjobb/oloah408/nerfs-test-pipeline/renders/"
folders = os.listdir(rendersPath)
folders.sort(reverse=True)

gammaValues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
gammaValues.sort(reverse=True)

for i in range(len(gammaValues)):
    if i >= len(folders):
        print("Not enough folders to match the number of gamma values.")
        break
    #get the nthn latest folder from the renders directory
    csvFilePath = os.path.join(rendersPath, folders[i], "test/rgb/quality_metrics.csv")
    print("Calculating average quality measures for cluster 0, 1 and 2 for gamma value: ", gammaValues[i])
    print("CSV file path: ", csvFilePath)
    cluster0, cluster1, cluster2 = calculateAverageQualityMeasuresGivenCluster(csvFilePath, scenePath, 3)

    print("Cluster 0: PSNR: ", cluster0[0], " SSIM: ", cluster0[1], " LPIPS: ", cluster0[2])
    print("Cluster 1: PSNR: ", cluster1[0], " SSIM: ", cluster1[1], " LPIPS: ", cluster1[2])
    print("Cluster 2: PSNR: ", cluster2[0], " SSIM: ", cluster2[1], " LPIPS: ", cluster2[2])

# Output the values to a file in the same directory as the CSV file
    outputFilePath = csvFilePath.split("quality_metrics.csv")[0] + "cluster_quality_metrics.txt"
    with open(outputFilePath, "w") as outputFile:
        outputFile.write("Cluster\t0\t1\t2\n")
        outputFile.write(f"Avg PSNR\t{cluster0[0]}\t{cluster1[0]}\t{cluster2[0]}\n")
        outputFile.write(f"Avg SSIM\t{cluster0[1]}\t{cluster1[1]}\t{cluster2[1]}\n")
        outputFile.write(f"Avg LPIPS\t{cluster0[2]}\t{cluster1[2]}\t{cluster2[2]}\n")
    outputFile.close()
    

#0 = dark, 1 = medium, 2 = light (riverview scene)
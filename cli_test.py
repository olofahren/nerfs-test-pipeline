import json
import os
import subprocess
from subprocess import check_output
import signal
import time
import re

from matplotlib import pyplot as plt
import numpy as np
import dataaug_test
import wandb
#from entropy_measurement import calculateAvgSceneEntropy
import extract_views
import gamma_correction
from histogram_matching import calculateAverageHistogram
import quality_metrics
import video_creator
import set_splits
import wandb
from PIL import Image

#variable inits
data_folder = []
datatype=[]
nerf_model=[]
viewer=[]
augmentationType = []
dataset = []
imageFiletype = []
crfNr = []
clipValue = []
gamma = []
splitSettings = []
testSetMode = []
additionalSettings = []
testSetFolder = []
std = []

#ns-train nerfbusters --data /home/exjobb/oloah408/nerfs-test-pipeline/data/blender/hotdog/ --pipeline.nerf-checkpoint-path /home/exjobb/oloah408/nerfs-test-pipeline/outputs/unnamed/nerfacto/2024-07-17_023902/nerfstudio_models/step-000029999.ckpt nerfstudio-data --eval-mode train-split-fraction


# WARNING: Using viewer will result in only the first run working 

# WARNING: The viewer frequently(almost always with certain NeRF models) causes
# some sort of dimension error when moving the view rapidly, causing a crash 


#-----------------------Specify settings for training-------------------
no_of_sessions = 1

#"--viewer.quit-on-train-completion=True"





#___________________________


# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching") #gamma/loss_gamma/histogrameq/histogram_matching/camera etc.
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.05)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.1)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.15)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.2)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.25)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.3)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.35)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.4)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.45)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.5)
# additionalSettings.append("")
# #--max-num-iterations=100

# data_folder.append("data/eyefultower/office_view2/images-jpeg-1k")
# imageFiletype.append("jpeg")
# datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
# nerf_model.append("nerfacto")
# viewer.append("wandb") #viewer+wandb
# augmentationType.append("histogram_matching")
# crfNr.append(2)
# clipValue.append(90)
# dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
# gamma.append(1.1)
# splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
# testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
# testSetFolder.append("data/eyefultower/office_view2/images-jpeg-1k/22")
# std.append(0.55)
# additionalSettings.append("")
# #--max-num-iterations=100


# ############################################################3
# #TEST RUN BELOW


data_folder.append("data/eyefultower/riverview/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("histogram_matching_HDR") #gamma/loss_gamma/histogrameq/histogram_matching/histogram_matching_HDR/camera etc.
crfNr.append(2)
clipValue.append(90)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.6)
splitSettings.append([0.9, 0.1, 0.1, 1000]) #train, eval, test, no of images in total // If testSetMode = "folder" -> set train + eval = 1
testSetMode.append("folder") #"random"/ "folder" /// If folder, test set is overridden with entire contents of folder
testSetFolder.append("data/eyefultower/riverview/images-1k/22")
std.append(0.4)
additionalSettings.append("")
#--max-num-iterations=100


#OBS!!!!! ÅTERSTÄLLNING AV BILDERNA ÄR AVSTÄNGD I restoreOriginalImagesEyefulTower i dataaug_test.py



#----------------------------------------------------------------------

print("Starting session with " + str(no_of_sessions) + " runs..." )
timeEstimate = no_of_sessions*39.3636
print("Estimated time left: Around " + str(timeEstimate) + " minutes.")
estimatedCompleteTime = time.time() + timeEstimate*60
print("Session estimated to be completed at " + time.strftime("%H:%M:%S", time.localtime(estimatedCompleteTime)))
time.sleep(4)

#Running training
for i in range(no_of_sessions):
        print("Training session " + str(i) +" with "+ nerf_model[i] + ". Data folder: " + data_folder[i])
        
        runName = "Dataset:"+dataset[i] + "_Model:" + nerf_model[i] + "_Augmentation:"+ augmentationType[i] + "_CRFnr:"+ str(crfNr[i]) + "_ClipValue"+ str(clipValue[i]) + "_Gamma:"+ str(gamma[i])

        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        os.environ['WANDB_NAME'] = runName
        os.environ['GAMMA_VALUE'] = str(1.0)
        #os.environ["WANDB_MODE"] = "disabled"
        
        #create a backup of the transforms file
        os.system("cp "+data_folder[i]+"/transforms.json "+data_folder[i]+"/transforms_copy.json")
        
        #/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/riverview/images-jpeg-1k/
        
        if dataset[i] == "eyefulTower":
                [trainSplitFileNames, 
                valSplitFileNames,
                testSplitFileNames] = set_splits.setSplit(data_folder[i]+"/transforms.json", testSetMode[i], splitSettings[i][0], splitSettings[i][1], splitSettings[i][2], splitSettings[i][3], testSetFolder[i])
                imagesToAugment = valSplitFileNames + trainSplitFileNames                
                
        
        #Fixing the training data by copying the images to preserve the original data
        # and applying data augmentations to the training data
        if augmentationType[i] == "gamma" or augmentationType[i] == "loss_gamma":
                dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i], crfNr[i], clipValue[i], gamma[i], testSplitFileNames, imagesToAugment)
        
        elif augmentationType[i] == "histogram_matching" or augmentationType[i] == "histogram_matching_HDR":
                 
                dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i], crfNr[i], clipValue[i], 0, testSplitFileNames, imagesToAugment, std[i])
        else: 
                dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i], crfNr[i], clipValue[i], 0, testSplitFileNames, imagesToAugment)
                
        
        time.sleep(1)
        
        command = ["ns-train", nerf_model[i], "--vis", viewer[i], datatype[i], "--data", data_folder[i], additionalSettings[i]]
        command = [part for part in command if part.strip()] #removing possible blank spaces
        
        print("Running command: ", ' '.join(command))
        
        process = subprocess.run(command)

        del os.environ['WANDB_NAME']
        
        #Restoring the original images
        if dataset[i] == "blender":
                print("Restoring original images for session " + str(i))
                dataaug_test.restoreOriginalImagesBlender("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train")
        elif dataset[i] == "eyefulTower": 
                print("Restoring original images for session " + str(i))
                dataaug_test.restoreOriginalImagesEyefulTower("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", imageFiletype[i])
                if augmentationType[i] != "histogram_matching_HDR":
                        avgHistOriginalImages = calculateAverageHistogram(dataaug_test.loadImagesWithFilenames("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "eyefulTower", imagesToAugment), 256, 100)

        
        
        
        
        #Rendering the test dataset from the trained nerf model
        runName = str(extract_views.getLatestFolderConsideringDataset(dataset[i], nerf_model[i]))
        
        extract_views.renderTestViews(nerf_model[i], dataset[i], data_folder[i])
        
        novelRenderedImages = dataaug_test.loadImagesWithFilenames("renders/"+runName+"/test/rgb", "other", testSplitFileNames)
        
        #Applying inverse transformations to the images if needed
        if augmentationType[i] == "gamma":
                gammaCorrectedNovelImages = gamma_correction.adjustGamma(1/gamma[i], novelRenderedImages)
        
                #save the gamma corrected images to the same folder
                print("Saving inverse gamma corrected images to renders/"+runName+"/test/rgb/...")
                for image in gammaCorrectedNovelImages:
                    image[1].save("renders/"+runName+"/test/rgb/"+image[0])
        elif augmentationType[i] == "histogrameq":
                # Extract image data for inverse histogram equalization
                image_data = [image[1] for image in novelRenderedImages]

                # Apply inverse histogram equalization
                inverseHisteqCorrectedImages = []
                for image in novelRenderedImages:
                        # Normalize image between 0 and 1
                        image_data = np.array(image[1])/255     
                        #________________________________________________FIX THIS
                        #normalized_image = (image[1] - np.min(image[1])) / (np.max(image[1]) - np.min(image[1]))
                        #print(f"Normalized image min: {np.min(normalized_image)}, max: {np.max(normalized_image)}")
                
                        # Apply inverse histogram equalization
                        corrected_image_data = np.interp(image_data, dataaug_test.histeqfunc, dataaug_test.bincenters)
                        #print(f"Corrected image data min: {np.min(corrected_image_data)}, max: {np.max(corrected_image_data)}")
                        
                        # Scale back to [0, 255]
                        corrected_image_data = (corrected_image_data * 255).astype(np.uint8)
                        corrected_image = Image.fromarray(corrected_image_data)
                        inverseHisteqCorrectedImages.append((image[0], corrected_image))

                # Save the inverse histogram equalized images to the same folder
                print("Saving inverse histogram equalized images to renders/"+runName+"/test/rgb/...")
                for image in inverseHisteqCorrectedImages:
                        image[1].save("renders/"+runName+"/test/rgb/"+image[0])

        elif augmentationType[i] == "histogram_matching":
                inverseHistogramMatchedImages = []
                image_data = [image[1] for image in novelRenderedImages]
                
                avgHistogramBeforeInverseTransform = calculateAverageHistogram(novelRenderedImages, 256, 100)
                
                for image in novelRenderedImages:
                        # Extract the image data
                        image_data = np.array(image[1])/255
                        
                        # Normalize image between 0 and 1
                        #normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
                        
                        
                        # Apply inverse histogram matching
                        xti = np.interp(image_data, dataaug_test.bc, dataaug_test.t_target)  # Mapping from Gaussian distribution to uniform
                        corrected_image_data = np.interp(xti, dataaug_test.t, dataaug_test.bc)  # Mapping from uniform distribution to original
                        
                        # Scale back to [0, 255]
                        corrected_image_data = (corrected_image_data * 255).astype(np.uint8)
                        corrected_image = Image.fromarray(corrected_image_data)
                        inverseHistogramMatchedImages.append((image[0], corrected_image))
                        
                        
                print("Saving inverse histogram matched images to renders/" + runName + "/test/rgb/...")
                for image in inverseHistogramMatchedImages:
                        image[1].save("renders/" + runName + "/test/rgb/" + image[0])
                
                
                
                
        else: 
                print("No inverse transformations applied.")
            
        print("Rendering training set video...")
        video_creator.createVideo("renders/"+runName+"/test/rgb", "renders/"+runName+"/test/rgb/"+nerf_model[i]+"_gamma_"+str(gamma[i])+".mp4")
        
        #Calculating the quality metrics
        if dataset[i] == "blender":
              quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test", "renders/"+runName+"/test/rgb/", gamma[i], dataset[i])
        elif dataset[i] == "eyefulTower":
                #copy the test files into a new folder in the same dir as the data.
                set_splits.makeTestSetFolder(testSplitFileNames, data_folder[i])
                if augmentationType[i] == "gamma":
                        quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test_images", "renders/"+runName+"/test/rgb/", gamma[i], dataset[i], "gamma")
                elif augmentationType[i] == "histogrameq":
                        #plt.plot(dataaug_test.avgHistEqImages)
                        #plt.savefig("renders/"+runName+"/test/rgb/avg_histogram_equalized_images.png")
                        #plt.close()
                        
                        
                        quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test_images", "renders/"+runName+"/test/rgb/", "histogrameq", dataset[i],"histogrameq")
                elif augmentationType[i] == "histogram_matching" or augmentationType[i] == "histogram_matching_HDR":
                        
                        images = dataaug_test.loadImagesWithFilenames("renders/"+runName+"/test/rgb", "other")
                        avgHist = calculateAverageHistogram(images, 256, 100)
                        #save the histogram to an image
                        plt.plot(avgHist)
                        plt.savefig("renders/"+runName+"/test/rgb/avg_histogram_novel_images.png")
                        plt.close()
                        
                        plt.plot(dataaug_test.avgHistAfterHistMatch)
                        plt.savefig("renders/"+runName+"/test/rgb/avg_histogram_matched_images.png")
                        plt.close()
                        
                        
                        
                        if augmentationType[i] != "histogram_matching_HDR":
                                
                                plt.plot(avgHistogramBeforeInverseTransform)
                                plt.savefig("renders/"+runName+"/test/rgb/avg_histogram_before_inverse_transform.png")
                                plt.close()
                        
                                plt.plot(avgHistOriginalImages)
                                plt.savefig("renders/"+runName+"/test/rgb/avg_histogram_original_images.png")
                                plt.close()

                        quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test_images", "renders/"+runName+"/test/rgb/", std[i], dataset[i],"histogram_matching")

                        
                        
                        
                elif augmentationType[i] == "none":
                        quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test_images", "renders/"+runName+"/test/rgb/", "none", dataset[i],"none")
                elif augmentationType[i] == "loss_gamma":
                        quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test_images", "renders/"+runName+"/test/rgb/", gamma[i], dataset[i], "loss_gamma")
                
        #Restore transforms.json
        print("Restoring original transforms.json file...")
        os.system("rm "+data_folder[i]+"/transforms.json")
        os.system("mv "+data_folder[i]+"/transforms_copy.json "+data_folder[i]+"/transforms.json")
        print("Training run " + str(i) + " quit.")

print("Session completed at " + time.strftime("%H:%M:%S", time.localtime(time.time())))
print("The session was predicted to be completed at " + time.strftime("%H:%M:%S", time.localtime(estimatedCompleteTime)))
averageTimePerSession = (time.time()-estimatedCompleteTime)/no_of_sessions
print("Average time per session: " + str(averageTimePerSession) + " seconds.")
print("Training session finished/failed!")


import json
import os
import subprocess
from subprocess import check_output
import signal
import time
import re
import dataaug_test
import wandb
import extract_views
import gamma_correction
import quality_metrics
import video_creator
import adjust_camera_pos

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
additionalSettings = []

#ns-train nerfbusters --data /home/exjobb/oloah408/nerfs-test-pipeline/data/blender/hotdog/ --pipeline.nerf-checkpoint-path /home/exjobb/oloah408/nerfs-test-pipeline/outputs/unnamed/nerfacto/2024-07-17_023902/nerfstudio_models/step-000029999.ckpt nerfstudio-data --eval-mode train-split-fraction


# WARNING: Using viewer will result in only the first run working 

# WARNING: The viewer frequently(almost always with certain NeRF models) causes
# some sort of dimension error when moving the view rapidly, causing a crash 

# mipnerf gets stuck and does not produce any results

#-----------------------Specify settings for training-------------------
no_of_sessions = 1

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#Session 0
data_folder.append("data/eyefultower/riverview/images-jpeg-1k")
imageFiletype.append("jpeg")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("viewer") #viewer+wandb
augmentationType.append("none")
crfNr.append(2)
clipValue.append(90)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1)
additionalSettings.append("")
#"--pipeline.datamanager.train-num-images-to-sample-from=100"

#Session 0
data_folder.append("data/eyefultower/riverview/images-jpeg-1k")
imageFiletype.append("jpeg")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("instant-ngp")
viewer.append("wandb") #viewer+wandb
augmentationType.append("none")
crfNr.append(2)
clipValue.append(90)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1)
additionalSettings.append("")
#"--pipeline.datamanager.train-num-images-to-sample-from=100"


#----------------------------------------------------------------------

#Running training
for i in range(no_of_sessions):
        print("Training session " + str(i) +" with "+ nerf_model[i] + ". Data folder: " + data_folder[i])
        runName = "Dataset:"+dataset[i] + "_Model:" + nerf_model[i] + "_Augmentation:"+ augmentationType[i] + "_CRFnr:"+ str(crfNr[i]) + "_ClipValue"+ str(clipValue[i]) + "_Gamma:"+ str(gamma[i])
                
        os.environ['WANDB_NAME'] = runName
        
        #Fixing the training data by copying the images to preserve the original data
        # and applying data augmentations to the training data
        
        dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i], crfNr[i], clipValue[i], gamma[i])

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
        
        
        
        #Rendering the test dataset from the trained nerf model
        runName = str(extract_views.getLatestFolderConsideringDataset(dataset[i], nerf_model[i]))
        
        
        if dataset[i] == "eyefulTower":
                testSetFileNames = []
                #extract the test set from the json file
                print("Extracting test set from json file...")
                data = json.load(open(data_folder[i]+"/transforms.json"))
                #extract all file names in the test_filenames section
                for filename in data['test_filenames']:
                        testSetFileNames.append(filename)
                        
        
        extract_views.renderTestViews(nerf_model[i], dataset[i])
        
        
        novelRenderedImages = dataaug_test.loadImagesWithFilenames("renders/"+runName+"/test/rgb")
        if gamma[i] != 1:
                gammaCorrectedNovelImages = gamma_correction.adjustGamma(1/gamma[i], novelRenderedImages)
        
                #save the gamma corrected images to the same folder
                print("Saving inverse gamma corrected images to renders/"+runName+"/test/rgb/...")
                for image in gammaCorrectedNovelImages:
                    image[1].save("renders/"+runName+"/test/rgb/"+image[0])
            
        print("Rendering training set video...")
        video_creator.createVideo("renders/"+runName+"/test/rgb", "renders/"+runName+"/test/rgb/"+nerf_model[i]+"_gamma_"+str(gamma[i])+".mp4")
        
        #Calculating the quality metrics
        if dataset[i] == "blender":
              quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test", "renders/"+runName+"/test/rgb/", gamma[i], dataset[i])
        elif dataset[i] == "eyefulTower":
                #NOTE: change this folder if test set is changed in transform.json
                quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/24", "renders/"+runName+"/test/rgb/", gamma[i], dataset[i])
        
        
        
        print("Training run " + str(i) + " quit.")


print("Training session finished/failed!")


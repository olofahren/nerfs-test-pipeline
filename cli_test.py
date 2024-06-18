import os
import subprocess
from subprocess import check_output
import signal
import time
import re
import dataaug_test
import wandb

#variable inits
data_folder = []
datatype=[]
nerf_model=[]
viewer=[]
augmentationType = []
dataset = []
imageFiletype = []


# WARNING: Using viewer will result in only the first run working 
# WARNING: The viewer frequently causes some sort of overflow when moving the view rapidly, causing a crash 

#-----------------------Specify settings for training-------------------
no_of_sessions = 1

#Session 0
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 1
data_folder.append("data/eyefultower/office1a/images-jpeg-2k")
imageFiletype.append("jpeg")
datatype.append("") #blender-data/colmap. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("none")
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management




#----------------------------------------------------------------------


#Applying data augmentations to the images

#Fixing the training data by copying the images to preserve the original data
# and applying data augmentations to the training data
for i in range(no_of_sessions):
    dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i])
    



#Running training
for i in range(no_of_sessions):
    print("Training session " + str(i) +" with "+ nerf_model[i] + ". Data folder: " + data_folder[i])
    print("Running command: " + "ns-train "+ nerf_model[i] +" --vis "+ viewer[i] +" "+ datatype[i] +" --data "+ data_folder[i])
    
    command = ["ns-train", nerf_model[i], "--vis", viewer[i], datatype[i], "--data", data_folder[i]]
    command = [part for part in command if part.strip()] #removing possible blank spaces
    
    #process = subprocess.run(command)
        
    #Restoring the original images
    if dataset[i] == "blender":
            print("Restoring original images for session " + str(i))
            dataaug_test.restoreOriginalImagesBlender("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", imageFiletype[i])
    elif dataset[i] == "eyefulTower": 
            print("Restoring original images for session " + str(i))
            dataaug_test.restoreOriginalImagesEyefulTower("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", imageFiletype[i])

    print("Training run " + str(i) + " quit.")


print("Training session finished/failed!")


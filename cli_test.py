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
crfNr = []
clipValue = []


# WARNING: Using viewer will result in only the first run working 

# WARNING: The viewer frequently(almost always with certain NeRF models) causes
# some sort of dimension error when moving the view rapidly, causing a crash 

# mipnerf gets stuck and does not produce any results in wandb 

#-----------------------Specify settings for training-------------------
no_of_sessions = 10

#Session 0
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(90)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 1
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(91)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 2
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(92)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 3
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(93)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 4
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(94)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 5
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(95)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 6
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(96)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 7
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(97)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 8
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(98)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management

#Session 9
data_folder.append("data/eyefultower/office1a/images-1k")
imageFiletype.append("exr")
datatype.append("") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("splatfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("camera")
crfNr.append(2)
clipValue.append(99)
dataset.append("eyefulTower") #eyefulTower, blender, colmap. For correct folder management




#----------------------------------------------------------------------


#Applying data augmentations to the images




#Running training
for i in range(no_of_sessions):
    print("Training session " + str(i) +" with "+ nerf_model[i] + ". Data folder: " + data_folder[i])
    runName = "Dataset:"+dataset[i] + "_Model:" + nerf_model[i] + "_Augmentation:"+ augmentationType[i] + "_CRFnr:"+ str(crfNr[i]) + "_ClipValue"+ str(clipValue[i])
        
    os.environ['WANDB_NAME'] = runName

    
    #Fixing the training data by copying the images to preserve the original data
    # and applying data augmentations to the training data
    
    dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i], crfNr[i], clipValue[i])

    
    print("Running command: " + "ns-train "+ nerf_model[i] +" --vis "+ viewer[i] +" "+ datatype[i] +" --data "+ data_folder[i])
    
    command = ["ns-train", nerf_model[i], "--vis", viewer[i], datatype[i], "--data", data_folder[i]]
    command = [part for part in command if part.strip()] #removing possible blank spaces
    
    process = subprocess.run(command)
    
    del os.environ['WANDB_NAME']

        
    #Restoring the original images
    if dataset[i] == "blender":
            print("Restoring original images for session " + str(i))
            dataaug_test.restoreOriginalImagesBlender("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train")
    elif dataset[i] == "eyefulTower": 
            print("Restoring original images for session " + str(i))
            dataaug_test.restoreOriginalImagesEyefulTower("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", imageFiletype[i])

    print("Training run " + str(i) + " quit.")


print("Training session finished/failed!")


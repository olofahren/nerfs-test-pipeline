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
gamma = []


# WARNING: Using viewer will result in only the first run working 

# WARNING: The viewer frequently(almost always with certain NeRF models) causes
# some sort of dimension error when moving the view rapidly, causing a crash 

# mipnerf gets stuck and does not produce any results

#-----------------------Specify settings for training-------------------
no_of_sessions = 5

#Session 0
data_folder.append("data/blender/drums")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(2)

#Session 0
data_folder.append("data/blender/drums")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.5)

#Session 0
data_folder.append("data/blender/drums")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1)

#Session 0
data_folder.append("data/blender/drums")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.5)

#Session 0
data_folder.append("data/blender/drums")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.1)


#----------------------------------------------------------------------

#Running training
for i in range(no_of_sessions):
    print("Training session " + str(i) +" with "+ nerf_model[i] + ". Data folder: " + data_folder[i])
    runName = "Dataset:"+dataset[i] + "_Model:" + nerf_model[i] + "_Augmentation:"+ augmentationType[i] + "_CRFnr:"+ str(crfNr[i]) + "_ClipValue"+ str(clipValue[i]) + "_Gamma:"+ str(gamma[i])
        
    os.environ['WANDB_NAME'] = runName
    
    #Fixing the training data by copying the images to preserve the original data
    # and applying data augmentations to the training data
    
    dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", dataset[i], augmentationType[i], crfNr[i], clipValue[i], gamma[i])

    time.sleep(3)
    
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


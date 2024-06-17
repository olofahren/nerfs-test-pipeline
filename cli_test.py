import os
import subprocess
from subprocess import check_output
import signal
import time
import dataaug_test

#variable inits
data_folder = []
datatype=[]
nerf_model=[]
viewer=[]


# WARNING: Using viewer will result in only the first run working 
# WARNING: The viewer frequently causes some sort of overflow when moving the view rapidly, causing a failure 

#-----------------------Specify settings for training-------------------
no_of_sessions = 1

#Session 1
data_folder.append("data/blender/hotdog")
datatype.append("blender-data") #blender/colmap etc
nerf_model.append("nerfacto")
viewer.append("viewer+wandb") #viewer+wandb

#Session 2 NOT IN USE 
data_folder.append("data/blender/drums")
datatype.append("blender-data") #blender/colmap etc
nerf_model.append("nerfacto")
viewer.append("wandb")

#----------------------------------------------------------------------


#Applying data augmentations to the images

#Fixing the training data by copying the images to preserve the original data
# and applying data augmentations to the training data
for i in range(no_of_sessions):
    dataaug_test.augmentImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train", "test", "val", "blur")
    



#Running training
for i in range(no_of_sessions):
    print("Training session " + str(i) +" with "+ nerf_model[i] + ". Data folder: " + data_folder[i])
    
    print("Running command: " + "ns-train "+ nerf_model[i] +" --vis "+ viewer[i] +" "+ datatype[i] +" --data "+ data_folder[i])
    
    command = ["ns-train", nerf_model[i], "--vis", viewer[i], datatype[i], "--data", data_folder[i]]
    #os.system("ns-train "+ nerf_model[i] +" --vis "+ viewer[i] +" "+ datatype[i] +" --data "+ data_folder[i])
    
    process = subprocess.run(command)
    #process_output = check_output(command)
    #print(process_output)


    print("Training session " + str(i) + " quit!!!")

os.system("Training finished/failed")

#Restoring the original images
for i in range(no_of_sessions):
    print("Restoring original images for session " + str(i))
    dataaug_test.restoreOriginalImages("~/oloah408/nerfs-test-pipeline/"+data_folder[i]+"/", "train")
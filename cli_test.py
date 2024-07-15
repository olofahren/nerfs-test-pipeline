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
no_of_sessions = 12

#Session 0


#Session 0
data_folder.append("data/blender/chair")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb1
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.5)

#___________________________----        

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.5)

# #Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.6)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.7)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.8)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(0.9)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.0)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.1)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.2)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.3)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.4)

#Session 0
data_folder.append("data/blender/mic")
imageFiletype.append("png")
datatype.append("blender-data") #blender-data/colmap, etc. Blank for EyefulTower
nerf_model.append("nerfacto")
viewer.append("wandb") #viewer+wandb1
augmentationType.append("gamma")
crfNr.append(2)
clipValue.append(90)
dataset.append("blender") #eyefulTower, blender, colmap. For correct folder management
gamma.append(1.5)


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
                
        #Rendering the test dataset from the trained nerf model
        runName = str(extract_views.getLatestFolder("outputs/unnamed/"+nerf_model[i]+"/"))
        
        extract_views.renderTestViews("outputs/unnamed/"+nerf_model[i]+"/"+runName+"/config.yml", nerf_model[i])
        
        novelRenderedImages = dataaug_test.loadImagesWithFilenames("renders/"+runName+"/test/rgb")
        gammaCorrectedNovelImages = gamma_correction.adjustGamma(1/gamma[i], novelRenderedImages)
        #save the gamma corrected images to the same folder
        print("Saving inverse gamma corrected images to renders/"+runName+"/test/rgb/...")
        for image in gammaCorrectedNovelImages:
            image[1].save("renders/"+runName+"/test/rgb/"+image[0])
            
        print("Rendering training set video...")
        video_creator.createVideo("renders/"+runName+"/test/rgb", "renders/"+runName+"/test/rgb/"+nerf_model[i]+"_gamma_"+str(gamma[i])+".mp4")
        
        #Calculating the quality metrics
        quality_metrics.getQualityMetricsFromFolder(data_folder[i]+"/test", "renders/"+runName+"/test/rgb/", gamma[i])
        
        
        
        
        print("Training run " + str(i) + " quit.")


print("Training session finished/failed!")


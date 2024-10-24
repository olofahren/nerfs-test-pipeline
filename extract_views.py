import os
import nerfstudio
import nerfstudio.data
import nerfstudio.data.dataparsers
import nerfstudio.data.dataparsers.base_dataparser

#ns-render dataset --load-config outputs/unnamed/nerfacto/2024-06-28_110459/config.yml --rendered-output-names rgb

#ns-render dataset --load-config outputs/unnamed/nerfacto/2024-06-28_110459/config.yml --output-path renders/--rendered-output-names rgb



def renderTestViews(nerfModel, dataset, dataFolder):
    print("Running command: ")
    
    if dataset == "blender":
        runName = str(getLatestFolder("outputs/unnamed/"+nerfModel+"/"))
        configPath = "outputs/unnamed/"+nerfModel+"/"+runName+"/config.yml"
        
        print("ns-render dataset --load-config " + configPath + " --output-path renders/"+ runName +" --rendered-output-names rgb")
        print("Rendering test views... This may take a while.")
        os.system("ns-render dataset --load-config " + configPath + " --output-path renders/"+ runName +" --rendered-output-names rgb")
    
    if dataset == "eyefulTower":
        #CHANGE THIS WHEN CHANGING THE DATASET OR DATA TYPE FROM/TO EXR/JPG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        runName = str(getLatestFolder("outputs/images-1k/"+nerfModel+"/"))
        configPath = "outputs/images-1k/"+nerfModel+"/"+runName+"/config.yml"
        
        #nerfstudio.data.dataparsers.base_dataparser.DataParser("nerfs-test-pipeline/outputs/images-jpeg-1k/instant-ngp/2024-08-10_161103/config.yml")  
        
        print("ns-render dataset --load-config " + configPath + " --output-path renders/"+ runName +" --rendered-output-names rgb --data "+ dataFolder +" --split test")
        print("Rendering test views... This may take a while.")
        os.system("ns-render dataset --load-config " + configPath + " --output-path renders/"+ runName +" --rendered-output-names rgb --data "+ dataFolder +" --split test")
    else:
        print("Invalid dataset in extract_views.py")
        
    print("Test views rendered successfully.")

def getLatestFolder(path):
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    folders.sort()
    return folders[-1]

def getLatestFolderConsideringDataset(dataset, nerfModel):
    if dataset == "blender":
        path = "outputs/unnamed/"+nerfModel+"/"
    elif dataset == "eyefulTower":
        #CHANGE THIS WHEN CHANGING THE DATASET OR DATA TYPE FROM/TO EXR/JPG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        path = "outputs/images-1k/"+nerfModel+"/"
        print("Path to latest folder: " + path)
    else:
        print("Invalid dataset in extract_views.py")
        return
    return getLatestFolder(path)
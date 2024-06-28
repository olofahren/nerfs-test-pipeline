import nerfstudio
import os

#ns-render dataset --load-config outputs/unnamed/nerfacto/2024-06-28_110459/config.yml --rendered-output-names rgb

#ns-render dataset --load-config outputs/unnamed/nerfacto/2024-06-28_110459/config.yml --output-path renders/--rendered-output-names rgb



def renderTestViews(configPath, nerfModel):
    print("Running command: ")
    print("ns-render dataset --load-config " + configPath + " --output-path renders/"+ str(getLatestFolder("outputs/unnamed/"+nerfModel+"/")) +" --rendered-output-names rgb")
    print("Rendering test views... This may take a while.")
    os.system("ns-render dataset --load-config " + configPath + " --output-path renders/"+ str(getLatestFolder("outputs/unnamed/"+nerfModel+"/")) +" --rendered-output-names rgb")
    print("Test views rendered successfully.")

def getLatestFolder(path):
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    folders.sort()
    return folders[-1]
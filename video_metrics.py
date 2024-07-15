import video_creator
import quality_metrics
import os


groundTruthVideoPath = "data/blender/drums/testground_truth.mp4"
datasetName = "drums"
trainingDate = "2024-07-01"


video_creator.createVideo("data/blender/"+datasetName+"/test", "data/blender/"+datasetName+"/testground_truth.mp4")
qualityMetrics = []

#get all folder in /renders
folders = os.listdir("renders")

folders = [folder for folder in folders if trainingDate in folder]





#search all text files in all sub folders of /renders
for folder in folders:
    files = os.listdir("renders/"+folder+"/test/rgb")
    #print(files)

    for file in files:
        if(file.endswith(".txt")):
            print("Found a text file in folder: ", folder)
            print("File: ", file)
            print("Path: ", "renders/"+folder+"/test/rgb/"+file)
            
            with open("renders/"+folder+"/test/rgb/"+file, "r") as f:
                lines = f.readlines()
                if(datasetName in lines[0]):
                    print("Dataset name found in text file: ", datasetName)
                    gammaValue = lines[1].split(": ")[1]
                    print("Gamma value", gammaValue)
                    
                    #get the path to the corresponding video file
                    videoFiles = os.listdir("renders/"+folder)
                    for videoFile in videoFiles:
                        if(videoFile.endswith(".mp4")):
                            degradedVideoPath = "renders/"+folder+"/test/rgb/"+videoFile
                            print("Found a video file in folder: ", folder)
                            print("File: ", videoFile)
                            print("Path: ", degradedVideoPath)

                            #calculate the quality metrics
                            quality = quality_metrics.getQualityMetricsVideo(groundTruthVideoPath, degradedVideoPath)
                            print("Quality metrics: ", quality)
                            qualityMetrics.append(quality)
                            break
                    break
                else:
                    print("Dataset name not found in text file: ", datasetName)
                    

for qualityMetric in qualityMetrics:
    print("Metric: "+qualityMetric +" Gamma: "+gammaValue + " Dataset: "+datasetName + "\n")

 
        





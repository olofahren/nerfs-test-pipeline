import json
import os
import numpy as np
import dataaug_test

def setSplit(splitFilePath, mode = "random", trainSplit = 0.8, valSplit= 0.1, testSplit=0.1, maxNumOfPics=1500, testFolder = "data/eyefultower/office1b/images-jpeg-1k/22"):
    """
    This function is used to set the split of the dataset eyefultower. The split is done based on the filenames in the json file.
    The filenames are shuffled and then split into train, validation and test sets based on the split ratio provided.
    
    Parameters:
    splitFilePath: str
        The path to the json file containing the filenames and their split.
    mode: str
        The mode of splitting the data. Currently only random is supported.
    trainSplit: float
        The ratio of the dataset to be used for training.
    valSplit: float
        The ratio of the dataset to be used for validation.
    testSplit: float
        The ratio of the dataset to be used for testing.
    maxNumOfPics: int
        The maximum number of pictures that are used for training, testing and evaluation.
        
    Returns:
    list
        A list containing the filenames for training, validation and test sets.
    """
    
    trainSplitFileNames = []
    valSplitFileNames = []
    testSplitFileNames = []
    
    
    data = json.load(open(splitFilePath))
    
    #Extracting the filenames from the json file
    for filename in data['test_filenames']:
            testSplitFileNames.append(filename)
            
    for filename in data['train_filenames']:
            trainSplitFileNames.append(filename)
            
    for filename in data['val_filenames']:
            valSplitFileNames.append(filename)
            
    allFileNames = trainSplitFileNames + valSplitFileNames + testSplitFileNames
    np.random.shuffle(allFileNames)
    allFileNames = allFileNames[:maxNumOfPics]

    
    if mode == "random":
        print("Only using " + str(maxNumOfPics) + " images from the dataset.")        
        trainSplitFileNames = allFileNames[:int(len(allFileNames)*trainSplit)]
        valSplitFileNames = allFileNames[int(len(allFileNames)*trainSplit):int(len(allFileNames)*(trainSplit+valSplit))]
        testSplitFileNames = allFileNames[int(len(allFileNames)*(trainSplit+valSplit)):]
    
    elif mode == "folder":
        print("Using folder " + testFolder + " as test set.")
        
        #remove all files from testSplitFileNames from allFileNames

        allFileNames = [i for i in allFileNames if i not in testSplitFileNames]   
             
        trainSplitFileNames = allFileNames[:int(len(allFileNames)*trainSplit)]
        valSplitFileNames = allFileNames[int(len(allFileNames)*trainSplit):int(len(allFileNames)*(trainSplit+valSplit))]
        
        testSplitFileNames = dataaug_test.loadImagesFilenames(testFolder)
        for i, filename in enumerate(testSplitFileNames):
            filename = filename.split('_')[0]+"/"+filename
            testSplitFileNames[i] = filename

        
        
        print("File name of TESTSPLIT IMAGE " + testSplitFileNames[0])
        

        
        
        
    #Writing the filenames back to the json file
    data['train_filenames'] = trainSplitFileNames
    data['val_filenames'] = valSplitFileNames
    data['test_filenames'] = testSplitFileNames
    
    with open(splitFilePath, 'w') as outfile:
        json.dump(data, outfile)
        
    #remove the leading path from the filenames
    trainSplitFileNames = [filename.split('/')[-1] for filename in trainSplitFileNames]
    testSplitFileNames = [filename.split('/')[-1] for filename in testSplitFileNames]
    valSplitFileNames = [filename.split('/')[-1] for filename in valSplitFileNames]
        
    return [trainSplitFileNames, valSplitFileNames, testSplitFileNames]

def makeTestSetFolder(testSplitFileNames, rootDataFolder):
    print("Creating test_images folder...")
    os.system("")
    os.system("mkdir "+ rootDataFolder +"/test_images")
                
    for filename in testSplitFileNames:
        folderNumber = filename.split('_')[0]
        #if the file ending is exr, change it to jpg
        if filename.endswith('.exr'):
            filename = filename[:-4] + ".jpg"
            
        os.system("cp " + rootDataFolder + "/"+folderNumber +"/"+filename +" "+rootDataFolder +"/test_images")
            
            
            
    
    
    
import json
import numpy as np

def setSplit(splitFilePath, mode = "random", trainSplit = 0.8, valSplit= 0.1, testSplit=0.1):
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
    
    if mode == "random":
        np.random.shuffle(allFileNames)

        trainSplitFileNames = allFileNames[:int(len(allFileNames)*trainSplit)]
        valSplitFileNames = allFileNames[int(len(allFileNames)*trainSplit):int(len(allFileNames)*(trainSplit+valSplit))]
        testSplitFileNames = allFileNames[int(len(allFileNames)*(trainSplit+valSplit)):]
        
        
    #Writing the filenames back to the json file
    data['train_filenames'] = trainSplitFileNames
    data['val_filenames'] = valSplitFileNames
    data['test_filenames'] = testSplitFileNames
    
    with open(splitFilePath, 'w') as outfile:
        json.dump(data, outfile)
        
    return [trainSplitFileNames, valSplitFileNames, testSplitFileNames]

            
            
            
    
    
    
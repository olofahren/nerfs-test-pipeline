import ctypes
import gc
import glob
import math
import os
import subprocess
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import camera_simulation.camera_sim as camera_sim
import camera_simulation.img_io as img_io
import shutil
import getpass
#from entropy_measurement import calculateAvgSceneEntropy
import gamma_correction
import histogram_equalization
from histogram_matching import calculateAverageHistogram, histMatch
from hdr_tonemapping_to_normal_distirubution import histMatchHDR, calculateAverageHistogramHDR, loadImagesWithFilenamesHDR, saveImagesEyefultowerEXR2JPEG

#------------------------------Settings------------------------------

#Folders

#Root folder for data e.g. "~/oloah408/nerfs-test-pipeline/data/blender/chair"
#root = "~/oloah408/nerfs-test-pipeline/data/blender/hotdog/"
#train_folder = "train"
#test_folder = "test"
#val_folder = "val" 

#--------------------------------------------------------------------


# -------------------------DATA HANDLING FUNCTIONS-------------------------
def loadImages(folder):
    images = []
    abs_folder_path = os.path.expanduser(folder)
    print("Loading images from folder: " + abs_folder_path)
    
    if os.path.exists(abs_folder_path):
        #print("Found data folder")
        pass
    else:
        print("Folder does not exist")
        print("Current working directory: " + os.getcwd())
        return images  # Exit early if folder doesn't exist

    try:
        for filename in os.listdir(abs_folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(abs_folder_path, filename)
                try:
                    with Image.open(img_path) as img:
                        images.append(img.copy())
                except IOError:
                    print("[loadImages]Error opening image " + filename)
    except Exception as e:
        print("Error accessing folder " + abs_folder_path + ": " + str(e))

    return images


def loadImagesFilenames(folder):
    filenames = []
    folder = os.path.expanduser(folder)

    print("Loading images with filenames from folder: " + folder)
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.exr', ".png", ".jpg", "jpeg")):
                img_path = os.path.join(root, filename)
                try:
                    filenames.append(filename)
                except IOError:
                    print("[loadImagesFilenames]Error opening image " + filename)
    return filenames    

def loadImagesWithFilenames(folder, dataset, includeList=[]):
    """
    Loads images with their filenames from a specified folder and dataset.

    Args:
        folder (str): The path to the folder containing the images.
        dataset (str): The name of the dataset to load. Currently supports "eyefulTower".

    Returns:
        list: A list of tuples, where each tuple contains the subfolder name and the filename of an image.
              If the folder does not exist, returns an empty list.

    Raises:
        IOError: If there is an error opening an image file.
    
    Example:
        images = loadImagesWithFilenames("~/data/images", "eyefulTower")
        for subfolder, filename in images:
            print(f"Subfolder: {subfolder}, Filename: {filename}")
    """
    
    images = []
    abs_folder_path = os.path.expanduser(folder)
    print("Loading images with filenames from folder: " + abs_folder_path)
    
    if os.path.exists(abs_folder_path):
        #print("Found data folder")
        pass
    else:
        print("Folder does not exist")
        print("Current working directory: " + os.getcwd())
        return images  # Exit early if folder doesn't exist

    if dataset == "eyefulTower":
        print("Loading images with filenames from Eyeful Tower dataset")
        try:
            #load all images within all subfolders with their respective filenames
            for root, dirs, files in os.walk(abs_folder_path):
                #do not load the images in original_images folder
                if "original_images" in root:
                    continue
                for filename in files:
                    if includeList:
                        #print("Include list provided")
                        if filename.endswith((".png", ".jpg", "jpeg")) and filename in includeList:
                            img_path = os.path.join(root, filename)
                            try:
                                with Image.open(img_path) as img:
                                    images.append((filename, img.copy()))
                            except IOError:
                                print("[loadImagesWithFilenames - eyefultower, includelist]Error opening image " + filename)
                            
                    else:
                        #print("Include list NOT provided")
                        if filename.endswith(('.exr', ".png", ".jpg", "jpeg")):
                            img_path = os.path.join(root, filename)
                            try:
                                with Image.open(img_path) as img:
                                    images.append((filename, img.copy()))
                            except IOError:
                                print("[loadImagesWithFilenames - eyefultower]Error opening image " + filename)
        except Exception as e:
            print("Error accessing folder " + abs_folder_path + ": " + str(e))
          
    else: 
        try:
            for filename in os.listdir(abs_folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(abs_folder_path, filename)
                    try:
                        with Image.open(img_path) as img:
                            images.append((filename, img.copy()))
                    except IOError:
                        print("[loadImagesWithFilenames - other dataset]Error opening image " + abs_folder_path)
        except Exception as e:
            print("Error accessing folder " + abs_folder_path + ": " + str(e))

    return images

def displayImage(image):
    # Display image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # No axes for this plot
    plt.show()

def saveImage(image, filename):
    #abs_filename = os.path.expanduser(filename)
    image.save(filename)

#Does something weird. Makes the training produce shit. Nothing visually wrong with the images, but the NeRF freaks out.
def saveImagesBlender(images, folder):
    # Save images in folder
    abs_folder_path = os.path.expanduser(folder)
    print("Saving images to folder " + folder)

    for i, image in enumerate(images):
        saveImage(image[1], abs_folder_path+ "/" + image[0])
        
def saveImagesEyefultowerJPEG(images, folder):
    # Save images in folder
    abs_folder_path = os.path.expanduser(folder)
    print("Saving images to folder " + folder)

    for i, image in enumerate(images):
        imageFolderNumber = image[0].split('_')[0]
        saveImage(image[1], abs_folder_path+ "/" + imageFolderNumber +"/" + image[0])

        
def saveImageEyefulTower(image, filename):
    # Save images in correct folder structure
    #print(images)
    #filename = os.path.expanduser(filename)
    #delete the .exr file from the folder
    if not os.path.exists(filename):
        print("File does not exist")
    else:
        print("Deleting "+filename)
        os.system("sudo rm "+filename)
    #removing the .exr extension from filename
    filename = filename[:-4]
    
    print("Saving image " + filename + ".png")
    img_io.writeLDR(image, filename + ".png") 

        
        
# -------------------------DATA HANDLING FUNCTIONS-------------------------
        
# -------------------------DATA AUGMENTATION FUNCTIONS-------------------------
def addBlur(image):
    # Add blur to image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
    return blurred_image

def addNoise(image):
    # Add noise to image
    noisy_image = image.copy()
    noisy_image = noisy_image + 0.1 * noisy_image.std() * np.random.random(noisy_image.shape)
    return noisy_image

# -------------------------DATA AUGMENTATION FUNCTIONS-------------------------

# -------------------------COPY FILES-------------------------
def copyOriginalImagesBlender(root, train_folder):
    user = getpass.getuser()
    os.system("sudo mkdir " + root + "original_images")
    os.system("sudo mkdir "+ root +"original_images/train")
    os.system("sudo cp "+ root + train_folder +"/* " + root + "original_images/train")
    subprocess.run(["sudo", "chown", "-R", user + ":" + user, root + "original_images/train"])
    
def copyOriginalImagesEyefulTower(root):
    user = getpass.getuser()
    print("Creating backup of original images...")
    root = os.path.expanduser(root)
    os.system("sudo mkdir " + root + "original_images")
    #check if folders already exist
    if not os.path.exists(root + "original_images/0"):
        for i in range(0,32):
            os.system("sudo cp -r "+ root + "/"+str(i) + " " + root + "original_images/"+str(i) )
            subprocess.run(["sudo", "chown", "-R", user + ":" + user, root + "original_images/"+str(i)])
# -------------------------COPY FILES-------------------------

# -------------------------RESTORE FILES-------------------------

def restoreOriginalImagesBlender(root, train_folder):
    root = os.path.expanduser(root)
    user = getpass.getuser()

    #Restore original images
    print("Restoring original images from " + root + "original_images/train to " + root + train_folder)
    #os.system("sudo cp "+ root + "original_images/train/* " + root + train_folder)
    #subprocess.run(["sudo", "chown", "-R", user + ":" + user, root + train_folder])

def restoreOriginalImagesEyefulTower(root, imageFiletype):
    user = getpass.getuser()
    root = os.path.expanduser(root)
    if imageFiletype == "exr":
        print("Not changing .jpg to .exr after restoring original images")
        #os.system("sudo sed -i 's/.jpg/.exr/g' "+root+"md5sums.txt")
        #os.system("sudo sed -i 's/.jpg/.exr/g' "+root+"transforms.json")
        
    #get number of folders in root
    num_folders = len([name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])
    
    print("NOT RESTORING ORIGINAL IMAGES")
    # for i in range(num_folders):
    #     print("Restoring original images from " + root + "original_images/"+str(i)+ " to " + root + str(i))
    #     os.system("sudo rm -r " + root + str(i))
    
    # #copy the entire contents of original_images to the current folder
    # os.system("sudo cp -r " + root + "original_images/*"+ " " + root)
    # subprocess.run(["sudo", "chown", "-R", user + ":" + user, root])
    
    #remove the original_images folder
    os.system("sudo rm -r " + root + "original_images")
    
# -------------------------RESTORE FILES-------------------------


# -------------------------AUGMENT IMAGES-------------------------

def augmentImages(root, train_folder, test_folder, val_folder, dataset, dataAugmentationType, cameraResponseFunction=False, clipValue=False, gamma = 1, testImagesFileNames = [], allUsableFiles =[], std=0.15):
    #if dataset used is BLENDER
    if dataset == "blender":
        copyOriginalImagesBlender(root, train_folder)
        print("Load from "+ root + train_folder)
        images_train = loadImagesWithFilenames(root + train_folder, "blender")
        if dataAugmentationType == "blur":
                print("Applying blur to images")
                augmented_images_train = [addBlur(image) for image in images_train]
                saveImagesBlender(augmented_images_train, root + train_folder)
        elif dataAugmentationType == "noise":
            print("Applying noise to images")
            augmented_images_train = [addNoise(image) for image in images_train]
            saveImagesBlender(augmented_images_train, root + train_folder)
        elif dataAugmentationType == "none":
            print("No data augmentation applied")
            augmented_images_train = images_train
            saveImagesBlender(augmented_images_train, root + train_folder)
            

        elif dataAugmentationType == "camera":
            print("Simulating camera response is not implemented for blender dataset, since no HDR images are available")
            return 0
        elif dataAugmentationType == "gamma":
            print("Applying gamma correction to images")
            augmented_images_train = gamma_correction.adjustGamma(gamma, images_train)
            saveImagesBlender(augmented_images_train, root + train_folder)
        else:
            print("Invalid data augmentation type")
            return 0
    
    
    #if dataset used is EYEFULTOWER
    elif dataset == "eyefulTower":
        global bc, t_target, t, avgHistAfterHistMatch
        print("Backing up original images")
        copyOriginalImagesEyefulTower(root)   
        #load images from all folders
        images_train = []
        augmented_images_train = []
        
        root = os.path.expanduser(root)

        
        # for i in range(1,9):
        #     images_train.extend(loadImages(root + str(i)))
        global histmatchfunc, bincenters
        
        if dataAugmentationType == "camera":        
            #REQUIRES HDR IMAGES TO FUNCTION PROPERLY
            
            #Loading images with filenames
            images_train = loadImagesFilenames(root)
            #print(images_train)
            
            length = len(images_train)
            
            #replace all .exr file extensions with .png to make nerfstudio happy
            os.system("sudo sed -i 's/.exr/.png/g' "+root+"md5sums.txt")
            os.system("sudo sed -i 's/.exr/.png/g' "+root+"transforms.json")
            
            for image in images_train:
                print("Reading "+root + image[0]+"/" + image)
                print("Progress: "+str(images_train.index(image))+"/"+str(length))
                img_source = root + image[0]+"/" + image
                
                if not os.path.exists(img_source):
                    print("File does not exist")
                    continue
                else:
                    if not clipValue:
                        clip = 97  # How many pixels that will not be saturated, for exposure tuning(3% of all pixels will be saturated, value over 1, clipped to 1)
                    else:
                        clip = clipValue
                        
                    cs = camera_sim.CameraSim()
                    print("Simulating camera response function")
                    H = img_io.readEXR(img_source)

                    exposure = 1/np.percentile(H,clip) # Exposure based on the clipping point
                    
                    if cameraResponseFunction:
                        sind = cameraResponseFunction
                    else:
                        sind = np.random.randint(0, cs.N)  # Random camera response function

                    L, crf = cs.capture(exposure*H, 'rand', sind, noise=False)
                                    
                    saveImageEyefulTower(L, img_source)
                print("_____________________________________________________________")

        elif dataAugmentationType == "gamma":
            
            training_images = loadImagesWithFilenames(root, "eyefulTower", allUsableFiles)
            print("Length of training images list: "+str(len(training_images)))
            
            
            

            
            print("Applying gamma correction to images")
            
            #split into batches of 1000 images to prevent memory issues
            batch_size = 1000
            num_batches = math.ceil(len(training_images)//batch_size) + 1
            print("Number of gamma correction batches: "+str(num_batches))
            for i in range(num_batches):
                print("Gamma correction batch: "+str(i))
                batch = training_images[i*batch_size:(i+1)*batch_size]
                augmented_images_train = gamma_correction.adjustGamma(gamma, batch)
                saveImagesEyefultowerJPEG(augmented_images_train, root)
                del augmented_images_train
                del batch
                gc.collect()
            
            
            
            
            
            #training_images_wo_filenames = [image[1] for image in training_images]
            
            del training_images
            gc.collect()
            
            #global averageSceneEntropy
            #averageSceneEntropy = calculateAvgSceneEntropy(training_images_wo_filenames)
            #print("Average scene entropy: "+str(averageSceneEntropy))
            
            #del training_images_wo_filenames
            #gc.collect()
            #augmented_images_train = gamma_correction.adjustGamma(gamma, training_images)
            
            #saveImagesEyefultowerJPEG(augmented_images_train, root)
            #return averageSceneEntropy
            
        elif dataAugmentationType == "loss_gamma":
            print("Applying gamma correction in the loss function with gamma value: "+str(gamma))
            os.environ['GAMMA_VALUE'] = str(gamma)

            
        elif(dataAugmentationType == "histogrameq"):
            print("Performing histogram equalization...")
            #load images
            training_images = loadImagesWithFilenames(root, "eyefulTower", allUsableFiles)
            
            #mmm ugly code, no no no
            global histeqfunc, bincenters
            
            training_images_wo_filenames = [image[1] for image in training_images]
            histeqfunc, bincenters = histogram_equalization.histeq(training_images_wo_filenames, 256)
            #print(histeqfunc)
            
            batch_size = 300
            num_batches = math.ceil(len(training_images)//batch_size) + 1
            print("Number of histogram equalisation application batches: "+str(num_batches))
            
            #apply histogram equalization to images
            print("Applying histogram equalization to images...")
            for i in range(num_batches):
                batch = training_images[i*batch_size:(i+1)*batch_size]
                
                #apply the histogram equalization function to the images
                augmented_images_train = []
                for image in batch:
                    # Normalize image between 0 and 1
                    #normalized_image = (image[1] - np.min(image[1])) / (np.max(image[1]) - np.min(image[1]))
                    scaled_image = np.array(image[1])/255
                    #print(f"Normalized image min: {np.min(normalized_image)}, max: {np.max(normalized_image)}")
                    
                    # Apply histogram equalization
                    augmented_image_data = np.interp(scaled_image, bincenters, histeqfunc)
                    #print(f"Augmented image data min: {np.min(augmented_image_data)}, max: {np.max(augmented_image_data)}")
                    
                    # Scale back to [0, 255]
                    augmented_image_data = (augmented_image_data * 255).astype(np.uint8)
                    augmented_image = Image.fromarray(augmented_image_data)
                    augmented_images_train.append((image[0], augmented_image))

                global avgHistEqImages
#                avgHistEqImages = calculateAverageHistogram(augmented_images_train, bincenters)
                
                saveImagesEyefultowerJPEG(augmented_images_train, root)
                del augmented_images_train
                del batch
                gc.collect()        
            
            del training_images
            gc.collect()
            
        elif dataAugmentationType == "histogram_matching":
            print("Performing histogram matching...")
            # Load images
            training_images = loadImagesWithFilenames(root, "eyefulTower", allUsableFiles)
            
            augmented_images_train, bc, t_target, t, avgHistAfterHistMatch = histMatch(training_images, std=std, bins=256, batch_size=100)
            
            
            saveImagesEyefultowerJPEG(augmented_images_train, root)
            del augmented_images_train    
            gc.collect()   
            
        elif dataAugmentationType == "histogram_matching_HDR":
            print("Performing histogram matching before quantization step...")
            
            training_images = loadImagesWithFilenamesHDR(root)
            #batch_size = 300
            #for i in range(0, len(training_images), batch_size):
            #batch = training_images[i:i + batch_size]
            
            #This also saves the images to folder
            augmented_images_train, bc, t_target, t, avgHistAfterHistMatch, h_target = histMatchHDR(training_images, std=std, bins=256, batch_size=50, save_folder=root)
            
            #saveImagesEyefultowerEXR2JPEG(augmented_images_train, root)
            del augmented_images_train
            del training_images
            #del batch
            ctypes.CDLL("libc.so.6").malloc_trim(0) 
            gc.collect()
        
            #change all .exr file extensions back to .png in transforms.json
            os.system("sudo sed -i 's/.exr/.jpg/g' "+root+"transforms.json")

            
        elif dataAugmentationType == "none":
            print("No data augmentation applied")
        else:
            print("Invalid data augmentation type")
            return 0
    else:
        print("Invalid dataset")
        return 0

    
    
#Deprecated    
    # #Creating and copying original images to preserve original data
    # if dataset == "blender":
    #     copyOriginalImagesBlender(root, train_folder)
    # elif dataset == "eyefulTower":
    #     copyOriginalImagesEyefulTower(root)   

    # #Loading images 
    # print("Load from "+ root + train_folder)
    # images_train = []
    # if dataset == "blender":
    #     images_train = loadImages(root + train_folder)
    # elif dataset == "eyefulTower":
    #     for i in range(1,9):
    #         images_train.extend(loadImages(root + str(i)))

    # #Applying augmentations
    # if dataAugmentationType == "blur":
    #     print("Applying blur to images")
    #     augmented_images_train = [addBlur(image) for image in images_train]
    # elif dataAugmentationType == "noise":
    #     print("Applying noise to images")
    #     augmented_images_train = [addNoise(image) for image in images_train]
    # elif dataAugmentationType == "none":
    #     print("No data augmentation applied")
    #     augmented_images_train = images_train
    # elif dataAugmentationType == "camera":
        
    #     #REQUIRES HDR IMAGES TO FUNCTION PROPERLY
    #     print("Simulating camera response function")
    #     #Loading images with filenames, overwriting images_train
    #     images_train = loadImagesWithFilenames(root + train_folder)
    #     for image in images_train:
    #         print("reading "+root + train_folder + "/" + image[1])
    #         img_source = root + train_folder + "/" + image[1]
    #         clip = 97  # How many pixels that will not be saturated, for exposure tuning

    #         cs = camera_sim.CameraSim()
    #         H = img_io.readEXR(img_source)

    #         exposure = 1/np.percentile(H,clip) # Exposure based on the clipping point
    #         sind = np.random.randint(0, cs.N)  # Random camera response function

    #         L, crf = cs.capture(exposure*H, 'rand', sind, noise=False)
    #         augmented_images_train.append(L)
    # else:
    #     print("Invalid data augmentation type")
    #     return 0

    # if dataset == "blender":
    #     saveImagesBlender(augmented_images_train, root + train_folder)
    # elif dataset == "eyefulTower":
    #     saveImagesEyefulTower(augmented_images_train, root)

# -------------------------AUGMENT IMAGES-------------------------





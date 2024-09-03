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
import gamma_correction

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
                    print("Error opening image " + filename)
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
                    print("Error opening image " + filename)
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
                        if filename.endswith(('.exr', ".png", ".jpg", "jpeg")) and filename in includeList:
                            img_path = os.path.join(root, filename)
                            try:
                                with Image.open(img_path) as img:
                                    images.append((filename, img.copy()))
                            except IOError:
                                print("Error opening image " + filename)
                    else:
                        if filename.endswith(('.exr', ".png", ".jpg", "jpeg")):
                            img_path = os.path.join(root, filename)
                            try:
                                with Image.open(img_path) as img:
                                    images.append((filename, img.copy()))
                            except IOError:
                                print("Error opening image " + filename)
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
                        print("Error opening image " + abs_folder_path)
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
    os.system("sudo cp "+ root + "original_images/train/* " + root + train_folder)
    subprocess.run(["sudo", "chown", "-R", user + ":" + user, root + train_folder])

def restoreOriginalImagesEyefulTower(root, imageFiletype):
    user = getpass.getuser()
    root = os.path.expanduser(root)
    if imageFiletype == "exr":
        os.system("sudo sed -i 's/.png/.exr/g' "+root+"md5sums.txt")
        os.system("sudo sed -i 's/.png/.exr/g' "+root+"transforms.json")
        
    #get number of folders in root
    num_folders = len([name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])
    
    for i in range(num_folders):
        print("Restoring original images from " + root + "original_images/"+str(i)+ " to " + root + str(i))
        os.system("sudo rm -r " + root + str(i))
    
    #copy the entire contents of original_images to the current folder
    os.system("sudo cp -r " + root + "original_images/*"+ " " + root)
    subprocess.run(["sudo", "chown", "-R", user + ":" + user, root])
    
    #remove the original_images folder
    os.system("sudo rm -r " + root + "original_images")
    
# -------------------------RESTORE FILES-------------------------


# -------------------------AUGMENT IMAGES-------------------------

def augmentImages(root, train_folder, test_folder, val_folder, dataset, dataAugmentationType, cameraResponseFunction=False, clipValue=False, gamma = 1, testImagesFileNames = [], allUsableFiles =[]):
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
        print("Backing up original images")
        copyOriginalImagesEyefulTower(root)   
        #load images from all folders
        images_train = []
        augmented_images_train = []
        
        root = os.path.expanduser(root)

        
        # for i in range(1,9):
        #     images_train.extend(loadImages(root + str(i)))

        
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
            
            del training_images
            gc.collect()
            #augmented_images_train = gamma_correction.adjustGamma(gamma, training_images)
            
            #saveImagesEyefultowerJPEG(augmented_images_train, root)
            
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





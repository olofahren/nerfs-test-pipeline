import os
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
import numpy as np

#------------------------------Settings------------------------------

#Folders

#Root folder for data e.g. "~/oloah408/nerfs-test-pipeline/data/blender/chair"
root = "~/oloah408/nerfs-test-pipeline/data/blender/hotdog/"
train_folder = "train"
test_folder = "test"
val_folder = "val" 

#--------------------------------------------------------------------


def loadImages(folder):
    images = []
    abs_folder_path = os.path.expanduser(folder)
    print("Loading images from folder: " + abs_folder_path)
    
    if os.path.exists(abs_folder_path):
        print("Folder exists")
    else:
        print("Folder does not exist")
        # For debugging: Print the current working directory
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

def displayImage(image):
    # Display image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # No axes for this plot
    plt.show()

def saveImage(image, filename):
    image.save(filename)

def saveImages(images, folder):
    # Save images in folder
    abs_folder_path = os.path.expanduser(folder)

    for i, image in enumerate(images):
        saveImage(image, os.path.join(abs_folder_path, "r_" + str(i) + ".png"))

def addBlur(image):
    # Add blur to image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
    return blurred_image


def addNoise(image):
    # Add noise to image
    noisy_image = image.copy()
    noisy_image = noisy_image + 0.1 * noisy_image.std() * np.random.random(noisy_image.shape)
    return noisy_image

def augmentImages(root, train_folder, test_folder, val_folder, dataAugmentationType):
    #Creating and copying original images to preserve original data
    os.system("sudo mkdir " + root + "original_images")
    os.system("sudo mkdir "+ root +"original_images/train")
    os.system("sudo cp "+ root + train_folder +"/* " + root + "original_images/train")

    #Loading images 
    print("Load from "+ root + train_folder)
    images_train = loadImages(root + train_folder)

    #Applying augmentations
    if dataAugmentationType == "blur":
        augmented_images_train = [addBlur(image) for image in images_train]
    elif dataAugmentationType == "noise":
        augmented_images_train = [addNoise(image) for image in images_train]
    elif dataAugmentationType == "none":
        augmented_images_train = images_train
    else:
        print("Invalid data augmentation type")
        return 0

    saveImages(augmented_images_train, root + train_folder)

def restoreOriginalImages(root, train_folder):
    #Restore original images
    print("Restoring original images from " + root + "original_images/train to " + root + train_folder)
    os.system("sudo cp "+ root + "original_images/train/* " + root + train_folder)


#Deprecated code, moved to functions above
#------------------------------------------------------------

#Handle image folders and preserve original data
#os.system("sudo mkdir " + root + "original_images")
#os.system("sudo mkdir "+ root +"original_images/train")
#os.system("sudo mkdir "+root + "original_images/test")
#os.system("sudo mkdir "+root + "original_images/val")

#os.system("sudo cp "+ root + train_folder +"/* " + root + "original_images/train")
#os.system("sudo cp "+ root + test_folder +"/* " + root + "original_images/test")
#os.system("sudo cp "+ root + val_folder +"/* " + root + "original_images/val")



#Loading images and applying augmentations
# print("Load from "+ root + train_folder)
# images_train = loadImages(root + train_folder)
#images_test = loadImages(root + test_folder)
#images_val = loadImages(root + val_folder)

# blurred_images_train = [addBlur(image) for image in images_train]
#blurred_images_test = [addBlur(image) for image in images_test]
#blurred_images_val = [addBlur(image) for image in images_val]

# saveImages(blurred_images_train, root + train_folder)
#saveImages(blurred_images_test, root + test_folder)
#saveImages(blurred_images_val, root + val_folder)

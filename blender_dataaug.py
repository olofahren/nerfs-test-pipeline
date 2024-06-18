import os
from PIL import Image, ImageFilter
import numpy as np


class blenderDataAugmentation:
    
    def loadImages(folder):
        images = []
        abs_folder_path = os.path.expanduser(folder)
        print("Loading images from folder: " + abs_folder_path)

        if os.path.exists(abs_folder_path):
            print("Found data folder")
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

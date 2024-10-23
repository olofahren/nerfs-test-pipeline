import cv2
import os


import cv2
import os

from camera_simulation import img_io


def createVideo(root_folder, video_name):
    exrFlag = False
    # Gather all images from the root folder and its subfolders
    images = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                if "normal" not in file and "depth" not in file:
                    images.append(os.path.join(dirpath, file))
            if file.endswith(".exr"):
                exrFlag = True
                images.append(os.path.join(dirpath, file)) # Add the exr images to the list of images   
    
    # If no images found, print a message and exit
    if not images:
        print(f"No suitable images found in {root_folder}.")
        return
    
    # Sort images
    images.sort()
    
    # Read the first image to get the frame size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' if 'mp4v' does not work
    video = cv2.VideoWriter(video_name, fourcc, 15, (width, height))  # Frame rate is set to 15

    # Write each image to the video
    if exrFlag:
        for image in images:
            exr_image = img_io.readEXR(image)
            # Normalize the EXR image to 8-bit
            exr_image_8bit = cv2.normalize(exr_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            jpg_image = cv2.cvtColor(exr_image_8bit, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
            video.write(jpg_image)    
    else:
        for image in images:
            video.write(cv2.imread(image))
        
    

    # Release the video writer and close all OpenCV windows
    cv2.destroyAllWindows()
    video.release()

#createVideo("renders/2024-07-02_110136/test/rgb", "renders/2024-07-02_110136/test/rgb/drums_gamma_0.5.mp4")

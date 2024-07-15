import cv2
import os


import cv2
import os

def createVideo(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg")]
    #remove all images that contains the string "normal" or "depth"
    images = [img for img in images if "normal" not in img and "depth" not in img]
    # Correctly sort the images by extracting the numeric part after 'r_'
    images.sort(key=lambda x: int(x.split('r_')[1].split('.')[0]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' if 'mp4v' does not work
    video = cv2.VideoWriter(video_name, fourcc, 15, (width, height))  # Frame rate is set to 10

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

#createVideo("renders/2024-07-02_110136/test/rgb", "renders/2024-07-02_110136/test/rgb/drums_gamma_0.5.mp4")

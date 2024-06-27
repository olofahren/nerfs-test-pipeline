import dataaug_test
import cv2
import numpy as np
from PIL import Image



#Only works with LDR images(JPEG, PNG, etc.)
def adjustGamma(gamma, images):
    gammaCorrectedImages = []
    
    for image in images:
        # Convert image to float, scale to range 0.0 - 1.0
        img_float = np.asarray(image[1], dtype=np.float32) / 255.0
        
        # Apply gamma correction
        img_corrected = np.power(img_float, gamma)
        
        # Scale back to range 0-255 and convert to uint8
        img_corrected = (img_corrected * 255).astype('uint8')
        
        # Convert numpy array back to PIL Image
        img_corrected_pil = Image.fromarray(img_corrected)
        
        # Create a new tuple with the updated image
        gammaCorrectedImage = (image[0], img_corrected_pil)
        
        gammaCorrectedImages.append(gammaCorrectedImage)
    
    return gammaCorrectedImages

        
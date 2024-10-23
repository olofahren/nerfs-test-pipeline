import gc
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

def calculateAverageHistogramHDR(images, bins=256, batch_size=100):
    H = np.zeros(bins)
    print("Calculating average histogram...")
    for image in images:
        #print(f"Calculating histogram for image {images.index(image) + 1}/{len(images)}", end='\r')
        image = np.array(image[1])
        h, _ = np.histogram(image, bins)
        H += h
        del h, image
        gc.collect()
    H = (H / np.sum(H))
    return H

def loadImagesWithFilenamesHDR(folder):
    folder = os.path.expanduser(folder)
    print("Loading EXR images...")
    images = []
    counter = 0
    for root, dirs, files in os.walk(folder):
        if "original_images" in root:
            continue
        for filename in files:
            print(f"Loading image {counter + 1}/{len(files)}", end='\r')
            counter += 1
            if filename.endswith('.exr'):
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append((filename, img))
                        del img
                        gc.collect()
                except IOError:
                    print(f"Error opening image {filename}")
    return images

def histMatchHDR(images, std=0.1, bins=256, batch_size=100, save_folder=None):
    print("Performing histogram matching on HDR images...")
    imagesFileNames = [image[0] for image in images]
    images = [image[1] for image in images]
    for i in range(len(images)):
        np.clip(images[i], 1e-6, None, out=images[i])
        np.log(images[i] + 1, out=images[i])
        np.clip(images[i], 0, np.percentile(images[i], 98), out=images[i])
    h = calculateAverageHistogramHDR(images, bins, batch_size)
    b = np.linspace(0, 1, bins + 1)
    bc = (b[1:] + b[:-1]) / 2
    h_target = np.exp(-0.5 * ((bc - 0.5) / std) ** 2)
    h_target = h_target / np.sum(h_target) * np.sum(h)
    t = np.cumsum(h) / np.sum(h)
    t_target = np.cumsum(h_target) / np.sum(h_target)
    print("Histogram matching...")
    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]
        batch_filenames = imagesFileNames[batch_start:batch_end]
        for idx, image in enumerate(batch_images):
            print(f"Matching histogram for image {batch_start + idx + 1}/{len(images)}", end='\r')
            matched_image = np.interp(np.interp(image, bc, t), t_target, bc)
            batch_images[idx] = (batch_filenames[idx], matched_image)
            del image, matched_image
            gc.collect()
        if save_folder:
            saveImagesEyefultowerEXR2JPEG(batch_images, save_folder)
        del batch_images, batch_filenames
        gc.collect()
    newAvgHist = calculateAverageHistogramHDR(images, bins, batch_size)
    return images, bc, t_target, t, newAvgHist, h_target

def saveImagesEyefultowerEXR2JPEG(images, folder):
    abs_folder_path = os.path.expanduser(folder)
    print(f"Saving images to folder {folder}")
    for image in images:
        normalized_image = (image[1] - np.min(image[1])) / (np.max(image[1]) - np.min(image[1])) * 255
        image_normalized = normalized_image.astype(np.uint8)
        imageFolderNumber = image[0].split('_')[0]
        save_path = os.path.join(abs_folder_path, imageFolderNumber)
        os.makedirs(save_path, exist_ok=True)
        image_filename = os.path.splitext(image[0])[0] + ".jpg"
        cv2.imwrite(os.path.join(save_path, image_filename), image_normalized)
        del normalized_image, image_normalized, image
        gc.collect()

# Example usage
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# folder = "/home/exjobb/oloah408/nerfs-test-pipeline/data/eyefultower/riverview/images-1k/"
# bins = 256
# images = loadImagesWithFilenamesHDR(folder)
# avgHistogram = calculateAverageHistogramHDR(images, bins)
# imageFileNames = [image[0] for image in images]
# images = [image[1] for image in images]
# images = [np.clip(image, 1e-6, None) for image in images]
# images = [np.log(image + 1) for image in images]
# images = [np.clip(image, 0, np.percentile(image, 98)) for image in images]
# avgHistogramLog = calculateAverageHistogramHDR(images, bins)
# images, bc, t_target, t, avgHistAfterHistMatch, h_target = histMatchHDR(images, std=0.1, bins=256, batch_size=100, save_folder="/path/to/save/folder")
# images = [(imageFileNames[i], images[i]) for i in range(len(images))]
# plt.plot(avgHistogramLog, label="Original")
# plt.plot(avgHistAfterHistMatch, label="After hist match")
# plt.plot(h_target, label="Target")
# plt.legend()
# plt.show()
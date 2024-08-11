import os
import sys
import cv2
import numpy as np
import lpips
import torch
import dataaug_test
import extract_views
import sewar
import PIL
import video_creator
#import video_metrics
import pyfvvdp
import time
import glob



def getQualityMetricsVideo(ref_file, TST_FILE):
    display_name = 'standard_fhd'
    media_folder = os.path.join(os.path.dirname(__file__), '..','example_media', 'aliasing')

    #ref_file = os.path.join()
    #TST_FILEs = glob.glob(os.path.join(media_folder, 'ferris-*-*.mp4'))

    print("Testing video quality for files: ", TST_FILE, " with reference file: ", ref_file)
    fv = pyfvvdp.fvvdp(display_name=display_name, heatmap=None)
    
    vs = pyfvvdp.fvvdp_video_source_file(TST_FILE, ref_file, display_photometry=display_name )

    start = time.time()
    Q_JOD_static, stats_static = fv.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(TST_FILE, Q_JOD_static, end-start) )
    
    return Q_JOD_static

def getQualityMetrics(I1, I2):
    '''
    :param I1: represents original image matrix
    :param I2: represents degraded image matrix
    :return: psnr, ssim, lpips score
    '''
    psnr = sewar.full_ref.psnr(I1, I2)
    ssim = sewar.full_ref.ssim(I1, I2)
    #lpips = sewar.full_ref.lpips(I1, I2)
    return psnr, ssim

def getQualityMetricsFromFolder(folder1, folder2, optionalMetaData, dataset):
    '''
    :param folder1: represents original image folder
    :param folder2: represents degraded image folder
    :param optionalMetaData: represents gamma value or any other optional metadata
    '''
    print("Computing quality metrics for images in folders: " + folder1 + " and " + folder2)
    print("This may take a while...")
    
    ssimAll = []
    psnrAll = []
    lpipsAll = []
    
    groundTruth = dataaug_test.loadImagesWithFilenames(folder1)
    degraded = dataaug_test.loadImagesWithFilenames(folder2)
    
    groundTruth.sort()
    degraded.sort()
    
    #Very ugly solution for specifically the blender dataset    
    groundTruth = [image for image in groundTruth if "normal" not in image[0] and "depth" not in image[0]]
        
    #creating a ground thruth video
    print("Rendering ground truth video...")
    absFolder1 = os.path.abspath(folder1)
    print("Rendering video from images in folder: ", absFolder1)
    video_creator.createVideo(absFolder1, absFolder1 + "ground_truth.mp4")
    
    #get path to the file that has ending .mp4 in folder2
    degradedVideoFiles = glob.glob(folder2 + "/*.mp4")
    degradedVideoPath = degradedVideoFiles[0]  # Take the first MP4 file found

    
    Q_JOD_static = getQualityMetricsVideo(absFolder1 + "ground_truth.mp4", degradedVideoPath)
    
    jod = str(Q_JOD_static)
        
    os.system("rm "+folder2+"quality_metrics.txt")

    for i in range(len(groundTruth)):
        print("\rProgress: ", i, "/", len(groundTruth), end="")
        sys.stdout.flush()        
        
        print(" Comparing images: ", groundTruth[i][0], " and ", degraded[i][0])
        
        I1 = np.array(groundTruth[i][1])
        #convert to 3 channel image from 4 channel image(removing the alpha channel)
        I1 = cv2.cvtColor(I1, cv2.COLOR_RGBA2RGB)
        I2 = np.array(degraded[i][1])
        
        
        psnr = sewar.full_ref.psnr(I1, I2)
        ssim, _ = sewar.full_ref.ssim(I1, I2)
        
        
        I1 = torch.from_numpy(I1)
        I2 = torch.from_numpy(I2)
        
        I1 = I1.float() / 255.0
        I2 = I2.float() / 255.0

        I1 = I1.unsqueeze(0).permute(0, 3, 1, 2)
        I2 = I2.unsqueeze(0).permute(0, 3, 1, 2)
        
        

        #-------suppress output from lpips--------
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                loss_fn_vgg = lpips.LPIPS(net='vgg')
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        #------------------------------------                
        lpips_val = loss_fn_vgg(I1, I2)
        
        #convert lpips tensor to float
        lpips_val = lpips_val.item()
        
        
        psnrAll.append(psnr)
        ssimAll.append(ssim)
        lpipsAll.append(lpips_val)
        
        #Save to text file
        with open(folder2+"quality_metrics.txt", "a") as file:
            file.write("Image: " + groundTruth[i][0] + "\n")
            file.write("PSNR: " + str(psnr) + "\n")
            file.write("SSIM: " + str(ssim) + "\n")
            file.write("LPIPS: " + str(lpips_val) + "\n")
            file.write("\n")    
            
    #Calculate average
    psnrAvg = sum(psnrAll)/len(psnrAll)
    ssimAvg = sum(ssimAll)/len(ssimAll)
    lpipsAvg = sum(lpipsAll)/len(lpipsAll)
    
    #write to top of file
    with open(folder2+"quality_metrics.txt", "r") as file:
        filedata = file.read()
    with open(folder2+"quality_metrics.txt", "w") as file:
        file.write("Dataset: " + folder1 + "\n")
        file.write("Gamma value: " + str(optionalMetaData) + "\n")
        file.write("Average PSNR: " + str(psnrAvg) + "\n")
        file.write("Average SSIM: " + str(ssimAvg) + "\n")
        file.write("Average LPIPS: " + str(lpipsAvg) + "\n")
        file.write("FovVideoVDP score: " +jod+ "\n") 
        file.write("\n")
        file.write(filedata)
        
    print("Quality metrics computed and saved to " + folder2 + "quality_metrics.txt")
    
    
    
    
    
    
    
import random
import numpy as np
from vision import *
import configparser
from tqdm import tqdm
from utility import *
import matplotlib.pyplot as plt
import os
import cv2
import glob
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from einops import rearrange

config = configparser.ConfigParser()
config.read('hw8config.txt')

def main():
    top_dir = config['PARAMETERS']['top_dir']
    data_dir = config['PARAMETERS']['data_dir']
    data_dir = os.path.join(top_dir, data_dir)
    out_dir = config['PARAMETERS']['output']
    image_paths = glob.glob(data_dir+'/*.jpg')
    # print(image_paths)
    num_images = len(image_paths)
    cp = readImgCV(data_dir+"/Pic_11.jpg")
    id_cp= image_paths.index('hw8images/Dataset1/Pic_11.jpg')
    print(id_cp)
    cp_edges = cannyEdge(cp)
    cp_lines = houghLines(cp_edges, 48)
    cpvl,cphl,type=filterLines(cp_lines)
    cpvl = refineLines(cpvl, "v", 8)
    cphl = refineLines(cphl, "h", 10)
    cp_corners = getCorners(cpvl, cphl)
    H = []
    Corners = []


    for i in range(num_images):
        img = readImgCV(image_paths[i])
        orig_img = np.copy(img)
        img_edges = cannyEdge(img)
        lines = houghLines(img_edges, 48)
        l1, l2, type = filterLines(lines)
        if type == "v":
            refined_l1 = refineLines(l1, "v", 8)
            # img = plotLines(refined_l1, img, (255,0,0))
            refined_l2 = refineLines(l2, "h", 10)
            # img = plotLines(refined_l2, img, (0,255,0))
            corners = getCorners(refined_l1, refined_l2)
            # plotPoints(corners, orig_img, mode="ch",color=(255,0,0))
        elif type == "h":
            refined_l1 = refineLines(l1, "h", 10)
            # img = plotLines(refined_l1, img, (0,255,0))
            refined_l2 = refineLines(l2, "v", 8)
            # img = plotLines(refined_l2, img, (255,0,0))
            corners = getCorners(refined_l2, refined_l1)
            # plotPoints(corners, orig_img, mode="ch",color=(255,0,0))
        Corners.append(corners)
        H.append(hmat_pinv(cp_corners, corners))
    w = omegaCalc(H)
    K = zhangK(w)
    R,T = zhangRT(H, K)
    comb_array = paramComb(R,T, K)
    # R,T,K = paramSep(comb_array, num_images)
    # Hcam = CameraCalibrationHomography(K, R, T)
    # reprojCorners = CameraReporjection(Hcam, cp_corners)

    error = costFunCameraCaleb(comb_array, cp_corners, Corners)
    max_err, mean_err, var_err = getError(error)
    print(max_err, mean_err, var_err)

    opt_comb_array = least_squares(costFunCameraCaleb, comb_array, method='lm', args=(cp_corners, Corners))
    opt_comb_array=opt_comb_array.x
    # opt_R,opt_T,opt_K = paramSep(opt_comb_array, num_images)
    # optimized_Hcam = CameraCalibrationHomography(opt_K, opt_R, opt_T)
    # opt_reprojCorners = CameraReporjection(optimized_Hcam, cp_corners)
    lm_error = costFunCameraCaleb(opt_comb_array, cp_corners, Corners)
    lm_max_err, lm_mean_err, lm_var_err = getError(lm_error)
    print(lm_max_err, lm_mean_err, lm_var_err)




    # for i in range(num_images):
    #     img = readImgCV(image_paths[i])
    #     # img = plotPoints(Corners[i], img, mode="ch", color=(255,0,255))
    #     img = plotPoints(reprojCorners[i], img, mode="ch", color=(255,255,0))
    #     img = plotPoints(opt_reprojCorners[i], img, mode="ch", color=(0,255,255))
    #     cv2show(img, "img")

    # reprojectCorners(Hcam, id_cp, Corners, image_paths, cp_corners)

        # houghName = "houghLines_"+str(i+1)+'.png'
        # save_img(houghName, out_dir, img)
        # name = "corners_"+str(i+1)+'.png'
        # save_img(name, out_dir, orig_img)



if __name__ == '__main__':
    main()
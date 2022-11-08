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
    base_file = config['PARAMETERS']['base_file']
    base_img = os.path.join(data_dir,base_file)
    image_paths = glob.glob(data_dir+'/*.jpg')
    # print(image_paths)
    num_images = len(image_paths)
    cp = readImgCV(base_img)
    id_cp= image_paths.index(base_img)
    print(id_cp)
    cp_edges = cannyEdge(cp)
    cp_lines = houghLines(cp_edges, 48)
    cpl1,cpl2,cptype=filterLines(cp_lines)
    if cptype == "v":
        refined_cpl1 = refineLines(cpl1, "v", 8)
        refined_cpl2 = refineLines(cpl2, "h", 10)
        cp_corners = getCorners(refined_cpl1, refined_cpl2)
    if cptype == "h":
        refined_cpl1 = refineLines(cpl1, "h", 10)
        refined_cpl2 = refineLines(cpl2, "v", 8)
        cp_corners = getCorners(refined_cpl2, refined_cpl1)
    # plotPoints(cp_corners, cp, mode="ch", color=(255, 255, 0))
    # cv2show(cp, "cp")
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
            img = plotLines(refined_l1, img, (255,0,0))
            refined_l2 = refineLines(l2, "h", 10)
            img = plotLines(refined_l2, img, (0,255,0))
            corners = getCorners(refined_l1, refined_l2)
            plotPoints(corners, orig_img, mode="ch",color=(255,0,0))
        elif type == "h":
            refined_l1 = refineLines(l1, "h", 10)
            img = plotLines(refined_l1, img, (0,255,0))
            refined_l2 = refineLines(l2, "v", 8)
            img = plotLines(refined_l2, img, (255,0,0))
            corners = getCorners(refined_l2, refined_l1)
            plotPoints(corners, orig_img, mode="ch",color=(255,0,0))
        # cv2show(img,"img")
        # cv2show(orig_img, "orig")
        Corners.append(corners)
        H.append(hmat_pinv(cp_corners, corners))
    w = omegaCalc(H)
    K = zhangK(w)
    R,T = zhangRT(H, K)
    comb_array = paramComb(R,T, K)
    error = costFunCameraCaleb(comb_array, cp_corners, Corners)
    max_err, mean_err, var_err = getError(error)
    print(max_err, mean_err, var_err)

    opt_comb_array = least_squares(costFunCameraCaleb, comb_array, method='lm', args=(cp_corners, Corners))
    opt_comb_array=opt_comb_array.x
    lm_error = costFunCameraCaleb(opt_comb_array, cp_corners, Corners)
    lm_max_err, lm_mean_err, lm_var_err = getError(lm_error)
    print(lm_max_err, lm_mean_err, lm_var_err)

    comb_array_rd = np.append(np.array([0,0]), comb_array)
    opt_comb_array_rd = least_squares(costFunCameraCaleb, comb_array_rd, method='lm', args=(cp_corners, Corners, True))
    opt_comb_array_rd = opt_comb_array_rd.x
    lm_error_rd = costFunCameraCaleb(opt_comb_array_rd, cp_corners, Corners, True)
    lm_max_err_rd, lm_mean_err_rd, lm_var_err_rd = getError(lm_error_rd)
    print(lm_max_err_rd, lm_mean_err_rd, lm_var_err_rd)

    R,T,K = paramSep(comb_array, num_images)
    Hcam = CameraCalibrationHomography(K, R, T)
    reprojCorners = CameraReporjection(Hcam, cp_corners)

    opt_R,opt_T,opt_K = paramSep(opt_comb_array, num_images)
    optimized_Hcam = CameraCalibrationHomography(opt_K, opt_R, opt_T)
    opt_reprojCorners = CameraReporjection(optimized_Hcam, cp_corners)

    opt_R_rd, opt_T_rd, opt_K_rd, k1_rd, k2_rd = paramSep(opt_comb_array_rd, num_images,True)
    optimized_Hcam_rd = CameraCalibrationHomography(opt_K_rd, opt_R_rd, opt_T_rd)
    opt_reprojCorners_rd = CameraReporjection(optimized_Hcam_rd, cp_corners)
    opt_reprojCorners_rd = radialDistort(opt_reprojCorners_rd,k1_rd, k2_rd,opt_K_rd[0,2],opt_K_rd[1,2])

    for i in range(num_images):
        img = readImgCV(image_paths[i])
        # img = plotPoints(Corners[i], img, mode="ch", color=(255,0,255))
        img = plotPoints(reprojCorners[i], img, mode="ch", color=(255,255,0))
        img = plotPoints(opt_reprojCorners[i], img, mode="ch", color=(0,255,255))
        img = plotPoints(opt_reprojCorners_rd[i], img, mode="ch", color=(255,0,255))
        cv2show(img, "img")



    reprojectCorners(Hcam, id_cp, Corners, image_paths, cp_corners)
    reprojectCorners(optimized_Hcam, id_cp, Corners, image_paths, cp_corners)
    reprojectCorners(optimized_Hcam_rd, id_cp, Corners, image_paths, cp_corners)

        # houghName = "houghLines_"+str(i+1)+'.png'
        # save_img(houghName, out_dir, img)
        # name = "corners_"+str(i+1)+'.png'
        # save_img(name, out_dir, orig_img)



if __name__ == '__main__':
    main()
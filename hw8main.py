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
    out_dir = os.path.join(out_dir, config['PARAMETERS']['data_dir'])
    base_file = config['PARAMETERS']['base_file']
    corners_path = config['PARAMETERS']['corners']
    hough_lines_path = config['PARAMETERS']['hough_lines']
    hough_lines_filtered_path = config['PARAMETERS']['hough_lines_filtered']
    proj_corners_path = config['PARAMETERS']['proj_corners']
    proj_corners_lm_path = config['PARAMETERS']['proj_corners_lm']
    proj_corners_lm_rd_path = config['PARAMETERS']['proj_corners_lm_rd']
    reproj_corners_path = config['PARAMETERS']['reproj_corners']
    reproj_corners_lm_path = config['PARAMETERS']['reproj_corners_lm']
    reproj_corners_lm_rd_path = config['PARAMETERS']['reproj_corners_lm_rd']
    canny_path = config['PARAMETERS']['canny']

    corners_path = os.path.join(out_dir, corners_path)
    hough_lines_path=os.path.join(out_dir,hough_lines_path)
    hough_lines_filtered_path=os.path.join(out_dir,hough_lines_filtered_path)
    proj_corners_path=os.path.join(out_dir,proj_corners_path)
    proj_corners_lm_path=os.path.join(out_dir,proj_corners_lm_path)
    proj_corners_lm_rd_path=os.path.join(out_dir,proj_corners_lm_rd_path)
    reproj_corners_path=os.path.join(out_dir,reproj_corners_path)
    reproj_corners_lm_path=os.path.join(out_dir,reproj_corners_lm_path)
    reproj_corners_lm_rd_path=os.path.join(out_dir,reproj_corners_lm_rd_path)
    canny_path=os.path.join(out_dir,canny_path)

    Color_lines = (191,23,23)
    Color_v_lines = (23,37,191)
    Color_h_lines = (214,208,19)
    Color_corners = (19,36,214)
    Color_proj_corners = (6,61,9)
    Color_proj_corners_lm = (145,7,33)
    Color_proj_corners_lm_rd = (124,145,4)
    Color_reproj_corners = (133,222,207)
    Color_reproj_corners_lm = (52,201,110)
    Color_reproj_corners_lm_rd = (167,181,9)

    print("Processing and Filtering corners for the Base images .....................................")

    base_img = os.path.join(data_dir,base_file)
    image_paths = glob.glob(data_dir+'/*.jpg')
    np_img_path = np.array((image_paths))
    print("Array of Image Paths:\n", np_img_path)
    num_images = len(image_paths)
    cp = readImgCV(base_img)
    id_cp= image_paths.index(base_img)
    print("Base Image Index: ", id_cp)
    cp_edges = cannyEdge(cp)
    cp_lines = houghLines(cp_edges, 48)
    cp_line_img = plotLines(cp_lines, cp, Color_lines)
    cpl1,cpl2,cptype=filterLines(cp_lines)
    if cptype == "v":
        refined_cpl1 = refineLines(cpl1, "v", 8)
        cp_line_img_ref = plotLines(refined_cpl1, cp, Color_v_lines)
        refined_cpl2 = refineLines(cpl2, "h", 10)
        cp_line_img_ref =plotLines(refined_cpl2, cp_line_img_ref, Color_h_lines)
        cp_corners = getCorners(refined_cpl1, refined_cpl2)
    if cptype == "h":
        refined_cpl1 = refineLines(cpl1, "h", 10)
        cp_line_img_ref =plotLines(refined_cpl1, cp, Color_h_lines)
        refined_cpl2 = refineLines(cpl2, "v", 8)
        cp_line_img_ref =plotLines(refined_cpl2, cp_line_img_ref, Color_v_lines)
        cp_corners = getCorners(refined_cpl2, refined_cpl1)

    cp_corner_img = plotPoints(cp_corners, cp, mode="ch", color=Color_corners)
    save_img("base_img_lines.jpg", out_dir, cp_line_img)
    save_img("base_img_lines_filtered.jpg", out_dir, cp_line_img_ref)
    save_img("base_img_corners.jpg", out_dir, cp_corner_img)

    print("................................................................................................")

    H = []
    Corners = []



    print("Processing Corners to Project from GT to Img.............")

    for i in trange(num_images):
        img = readImgCV(image_paths[i])
        img_name = image_paths[i].split('/')[-1]
        img_edges = cannyEdge(img)
        lines = houghLines(img_edges, 48)
        orig_lines = plotLines(lines, img, Color_lines)
        l1, l2, type = filterLines(lines)
        if type == "v":
            refined_l1 = refineLines(l1, "v", 8)
            l_img = plotLines(refined_l1, img, Color_v_lines)
            refined_l2 = refineLines(l2, "h", 10)
            l_img = plotLines(refined_l2, l_img, Color_h_lines)
            corners = getCorners(refined_l1, refined_l2)
            p_img = plotPoints(corners, img, mode="ch",color=Color_corners)
        elif type == "h":
            refined_l1 = refineLines(l1, "h", 10)
            l_img = plotLines(refined_l1, img,Color_h_lines )
            refined_l2 = refineLines(l2, "v", 8)
            l_img = plotLines(refined_l2, l_img, Color_v_lines)
            corners = getCorners(refined_l2, refined_l1)
            p_img = plotPoints(corners, img, mode="ch",color=Color_corners)

        save_img(img_name, hough_lines_path, orig_lines)
        save_img(img_name, hough_lines_filtered_path, l_img)
        save_img(img_name, corners_path, p_img)
        save_img(img_name, canny_path, img_edges)
        Corners.append(corners)
        H.append(hmat_pinv(cp_corners, corners))
    w = omegaCalc(H)
    K = zhangK(w)
    R,T = zhangRT(H, K)
    comb_array = paramComb(R,T, K)
    error = costFunCameraCaleb(comb_array, cp_corners, Corners)
    max_err, mean_err, var_err = getError(error)
    print("Values of corner projection (max, mean, var) error without LM and RD: ", max_err, mean_err, var_err)
    print("................................................................................................")

    opt_comb_array = least_squares(costFunCameraCaleb, comb_array, method='lm', args=(cp_corners, Corners))
    opt_comb_array=opt_comb_array.x
    lm_error = costFunCameraCaleb(opt_comb_array, cp_corners, Corners)
    lm_max_err, lm_mean_err, lm_var_err = getError(lm_error)
    print("Values of corner projection (max, mean, var) error with LM and without RD: ", lm_max_err, lm_mean_err, lm_var_err)
    print("................................................................................................")

    comb_array_rd = np.append(np.array([0,0]), comb_array)
    opt_comb_array_rd = least_squares(costFunCameraCaleb, comb_array_rd, method='lm', args=(cp_corners, Corners, True))
    opt_comb_array_rd = opt_comb_array_rd.x
    lm_error_rd = costFunCameraCaleb(opt_comb_array_rd, cp_corners, Corners, True)
    lm_max_err_rd, lm_mean_err_rd, lm_var_err_rd = getError(lm_error_rd)
    print("Values of corner projection (max, mean, var) error with LM and RD: ", lm_max_err_rd, lm_mean_err_rd, lm_var_err_rd)
    print("................................................................................................")

    R,T,K = paramSep(comb_array, num_images)
    Hcam = CameraCalibrationHomography(K, R, T)
    reprojCorners = CameraReporjection(Hcam, cp_corners)
    print("Parameters before LM and RD ....................................................................")
    print("Intrinsic Camera parameters before LM and RD:\n", np.array(K))
    print("Rotation matrix before LM and RD:\n", np.array(R))
    print("Translation matrix before LM and RD:\n", np.array(T))
    print("................................................................................................")

    opt_R,opt_T,opt_K = paramSep(opt_comb_array, num_images)
    optimized_Hcam = CameraCalibrationHomography(opt_K, opt_R, opt_T)
    opt_reprojCorners = CameraReporjection(optimized_Hcam, cp_corners)
    print("Parameters with LM and without RD ....................................................................")
    print("Intrinsic Camera parameters with LM and without RD:\n", np.array(opt_K))
    print("Rotation matrix before with LM and without RD:\n", np.array(opt_R))
    print("Translation matrix before with LM and without RD:\n", np.array(opt_T))
    print("................................................................................................")

    opt_R_rd, opt_T_rd, opt_K_rd, k1_rd, k2_rd = paramSep(opt_comb_array_rd, num_images,True)
    optimized_Hcam_rd = CameraCalibrationHomography(opt_K_rd, opt_R_rd, opt_T_rd)
    opt_reprojCorners_rd = CameraReporjection(optimized_Hcam_rd, cp_corners)
    opt_reprojCorners_rd = radialDistort(opt_reprojCorners_rd,k1_rd, k2_rd,opt_K_rd[0,2],opt_K_rd[1,2])
    print("Parameters with LM and RD ....................................................................")
    print("k1 and K2 for radial distortions:\n", np.array([k1_rd, k2_rd]))
    print("Intrinsic Camera parameters with LM and RD:\n", np.array(opt_K_rd))
    print("Rotation matrix with LM and RD:\n", np.array(opt_R_rd))
    print("Translation matrix with LM and RD:\n", np.array(opt_T_rd))
    print("................................................................................................")

    print("Projecting the corners from GT to images ........")
    for i in trange(num_images):
        img = readImgCV(image_paths[i])
        img_name = image_paths[i].split('/')[-1]
        proj_img = plotPoints(reprojCorners[i], img, mode="ch", color=Color_proj_corners)
        proj_img_lm = plotPoints(opt_reprojCorners[i], img, mode="ch", color=Color_proj_corners_lm)
        proj_img_lm_rd = plotPoints(opt_reprojCorners_rd[i], img, mode="ch", color=Color_proj_corners_lm_rd)
        save_img(img_name, proj_corners_path, proj_img)
        save_img(img_name, proj_corners_lm_path, proj_img_lm)
        save_img(img_name, proj_corners_lm_rd_path, proj_img_lm_rd)
    print("................................................................................................")

    print("Re-Projecting the corners from images to GT ........")
    reproj_corners = reprojectCorners(Hcam, id_cp, Corners)
    reproj_corners_lm = reprojectCorners(optimized_Hcam, id_cp, Corners)
    reproj_corners_lm_rd = reprojectCorners(optimized_Hcam_rd, id_cp, Corners)
    img = readImgCV(image_paths[id_cp])
    for i in trange(num_images):
        img_name = image_paths[i].split('/')[-1]
        reproj_img = plotPoints(reproj_corners[i], img, mode="ch", color=Color_reproj_corners)
        reproj_img_lm = plotPoints(reproj_corners_lm[i], img, mode="ch", color=Color_reproj_corners_lm)
        reproj_img_lm_rd = plotPoints(reproj_corners_lm_rd[i], img, mode="ch", color=Color_reproj_corners_lm_rd)
        save_img(img_name, reproj_corners_path, reproj_img)
        save_img(img_name, reproj_corners_lm_path, reproj_img_lm)
        save_img(img_name, reproj_corners_lm_rd_path, reproj_img_lm_rd)

    print("................................................................................................")



if __name__ == '__main__':
    main()
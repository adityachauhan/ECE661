import configparser
import glob
import os

from vision import *
import argparse
config = configparser.ConfigParser()
config.read('hw9config.txt')
import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

if __name__=='__main__':
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'],config['PARAMETERS']['data_dir'])
    print(x_img_path)
    resize_img_path = os.path.join(config['PARAMETERS']['top_dir'],config['PARAMETERS']['data_dir'])
    img_paths = glob.glob(x_img_path+'/box*.jpeg')
    print(img_paths)
    num_imgs = len(img_paths)
    for i in range(num_imgs):
        img=readImgCV(img_paths[i])
        img = image_resize(img, width=1200, height=1600)
        imgname = img_paths[i].split('/')[-1]
        print(imgname)
        cv2show(img, "img")
        save_img_v2(imgname, resize_img_path,img)

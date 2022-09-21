import configparser
import os

from vision import *
import argparse
config = configparser.ConfigParser()
config.read('hw3config.txt')
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
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    img=cv2.imread(x_img_path)
    img = image_resize(img, width=600)
    # img_path = x_img_path
    cv2.imwrite(x_img_path, img)
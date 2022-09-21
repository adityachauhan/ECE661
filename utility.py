import configparser
import os

from vision import *
import argparse
config = configparser.ConfigParser()
config.read('hw3config.txt')

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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
def image_viewer(img):
    plt.imshow(img)
    plt.show()
## this click_event taken from
# https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/?fbclid=IwAR0c9Mvns8fNWDL0fWF6dFEmLxnUjRl_svnm8tdLO87GS7ncDbuM56mUa6U
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,',', y)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' +str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        cv2.imshow('img', img)



if __name__ == "__main__":

    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    img = cv2.imread(x_img_path)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
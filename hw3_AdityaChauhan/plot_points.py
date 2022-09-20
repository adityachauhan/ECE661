import configparser
import os

from vision import *
import argparse
config = configparser.ConfigParser()
config.read('hw3config.txt')

x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
top_dir= config['PARAMETERS']['top_dir']
img_name= config['PARAMETERS']['x_path']
img_name = img_name.split('.')[0]
img_name_dotted=img_name+"_dotted.jpg"
x_img = readImgCV(x_img_path)

ho, wo, co = x_img.shape
print(ho,wo,co)
ol_pts1 = config['PARAMETERS']['ol_pts1']
ol_pts2 = config['PARAMETERS']['ol_pts2']
ol_pts3 = config['PARAMETERS']['ol_pts3']
ol_pts4 = config['PARAMETERS']['ol_pts4']
ol_pts5 = config['PARAMETERS']['ol_pts5']

ol_pts1 = str2np(ol_pts1)
ol_pts2 = str2np(ol_pts2)
ol_pts3 = str2np(ol_pts3)
ol_pts4 = str2np(ol_pts4)
ol_pts5 = str2np(ol_pts5)

color = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255)]
pts_array = np.array((ol_pts1, ol_pts2, ol_pts3, ol_pts4, ol_pts5))
for pts in range(len(pts_array)):
    for i in range(len(pts_array[pts])):
        cv2.circle(x_img, (pts_array[pts][i][0], pts_array[pts][i][1]), radius=4, color=color[pts], thickness=-1)

img_path = os.path.join(top_dir, img_name_dotted)
x_img=cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
cv2.imwrite(img_path, x_img)
import configparser
import os

from vision import *

config = configparser.ConfigParser()
config.read('hw3config.txt')

def image_viewer(img):
    plt.imshow(img)
    plt.show()

def main():
    x_img_path = os.path.join(config['PARAMETERS']['top_dir'], config['PARAMETERS']['x_path'])
    x_img = readImgCV(x_img_path)
    image_viewer(x_img)

if __name__ == "__main__":
    main()
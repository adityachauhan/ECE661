import configparser
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from vision import *

config = configparser.ConfigParser()
config.read('config.txt')

def main():
    output = config['PARAMETERS']['output']
    p1 = os.path.join(config['PARAMETERS']['top_dir'],'card1.jpeg')
    img1 = readImgCV(p1)
    h1,w1,c1 = img1.shape
    homo_mode = config['PARAMETERS']['homo_mode']
    x_pts1 = config['PARAMETERS']['card_pts1']
    x_pts2 = config['PARAMETERS']['card_pts2']
    x_pts3 = config['PARAMETERS']['card_pts3']
    x_pts1=str2np(x_pts1)
    x_pts2=str2np(x_pts2)
    x_pts3=str2np(x_pts3)

    vision12 = Vision(x_pts1, x_pts2)
    h12 = vision12.calc_homograpy(homo_mode)

    vision23 = Vision(x_pts2, x_pts3)
    h23 = vision23.calc_homograpy(homo_mode)

    h13 = np.matmul(h23, h12)

    empty_img=np.zeros((h1,w1,c1), np.uint8)
    empty_img[:,:,:]=img1[10][10]

    l12 = np.cross(cvrt2homo(x_pts1[0]), cvrt2homo(x_pts1[1]))
    l24 = np.cross(cvrt2homo(x_pts1[1]), cvrt2homo(x_pts1[3]))
    l43 = np.cross(cvrt2homo(x_pts1[3]), cvrt2homo(x_pts1[2]))
    l31 = np.cross(cvrt2homo(x_pts1[2]), cvrt2homo(x_pts1[0]))
    for c in tqdm(range(w1)):
        for r in range(h1):
            X = np.array((c, r, 1))
            if np.dot(np.transpose(l12), X) >= 0 and np.dot(np.transpose(l24), X) >= 0 and np.dot(np.transpose(l43),X) >= 0 and np.dot(np.transpose(l31), X) >= 0:
                X_prime = np.matmul(h13, X)
                X_prime = X_prime / X_prime[2]
                X_prime = X_prime.astype(np.int32)
                if X_prime[1] < h1 and X_prime[0] < w1:
                    empty_img[X_prime[1]][X_prime[0]] = img1[r][c]

    # plt.imshow(empty_img)
    # plt.show()
    save_img('task1_part2.jpg', output, empty_img)



if __name__ == '__main__':
    main()
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from einops import rearrange
from tqdm import trange, tqdm
from scipy.signal import convolve2d
import sys

def readImgCV(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cvrt2homo(pt):
    return np.append(pt, 1)



def str2np(s):
    x_prime_pts = s.split(',')
    x_prime_pts = [int(val) for val in x_prime_pts]
    x_prime_pts = np.array(x_prime_pts)
    x_prime_pts = rearrange(x_prime_pts, '(c h)-> c h ', c=4, h=2)
    return x_prime_pts

def save_img(name, path, img):
    img_path = os.path.join(path, name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, img)

def xpeqhx(x,y,h):
    X = np.array((x, y, 1))
    X_prime = np.matmul(h, X)
    X_prime = X_prime / X_prime[2]
    # X_prime = X_prime.astype(int)
    return X_prime

def point2point(x,h):
    X = np.array((x[0], x[1], 1))
    X_prime = np.matmul(h, X)
    X_prime = X_prime/X_prime[2]
    X_prime = X_prime.astype(int)
    X_prime = np.array([X_prime[0], X_prime[1]])
    return X_prime
def vanishing_line_homography(x_pts):
    l12 = np.cross(cvrt2homo(x_pts[0]), cvrt2homo(x_pts[1]))
    l34 = np.cross(cvrt2homo(x_pts[2]), cvrt2homo(x_pts[3]))
    l31 = np.cross(cvrt2homo(x_pts[2]), cvrt2homo(x_pts[0]))
    l42 = np.cross(cvrt2homo(x_pts[3]), cvrt2homo(x_pts[1]))
    vpt1 = np.cross(l12, l34)
    vpt1 = vpt1 / vpt1[2]
    vpt2 = np.cross(l31, l42)
    vpt2 = vpt2 / vpt2[2]
    vl = np.cross(vpt1, vpt2)
    h = np.zeros((3,3))
    h[0][0] = 1
    h[1][1] = 1
    h[2][0] = vl[0]/vl[2]
    h[2][1] = vl[1]/vl[2]
    h[2][2] = vl[2]/vl[2]
    return h, vl

def primePtsP2P(pts):
    X=np.ones((len(pts),2))
    # print(X)
    X[0,0]=pts[0,0]
    X[0,1]=pts[0,1]
    X[1,0]=pts[1,0]
    X[1,1]=pts[0,1]
    X[2,0]=pts[0,0]
    X[2,1]=pts[2,1]
    X[3,0]=pts[1,0]
    X[3,1]=pts[2,1]
    return X
def make_line_hc(pts1, pts2):
    l = np.cross(cvrt2homo(pts1), cvrt2homo(pts2))
    return l

class Vision:
    def __init__(self, x, x_prime, homo_mat_size=3):
        self.x = x
        self.x_prime = x_prime
        self.h = np.ones((homo_mat_size,homo_mat_size))
        self.h_mat_size = homo_mat_size

    def calc_homograpy(self, homo_mode):
        if homo_mode=='projective':
            A = np.ones((len(self.x)*2,len(self.x)*2))
            C = np.ones(len(self.x)*2)
            for i in range(len(self.x)):
                A[2*i] = np.array([self.x[i][0], self.x[i][1], 1, 0, 0, 0, -self.x[i][0]*self.x_prime[i][0], -self.x[i][1]*self.x_prime[i][0]])
                A[(2*i)+1] = np.array([0, 0, 0, self.x[i][0], self.x[i][1], 1, -self.x[i][0]*self.x_prime[i][1], -self.x[i][1]*self.x_prime[i][1]])
                C[2*i] = self.x_prime[i][0]
                C[(2*i)+1] = self.x_prime[i][1]
            Ainv = np.linalg.inv(A)
            B = np.dot(Ainv, C)
            B = np.append(B, 1)
            self.h = rearrange(B, '(c h)-> c h',c=self.h_mat_size, h=self.h_mat_size)
        if homo_mode=='affine':
            A = np.ones(((len(self.x)-1) * 2, (len(self.x) -1)* 2))
            C = np.ones((len(self.x) -1) * 2)
            for i in range(len(self.x)-1):
                A[2 * i] = np.array([self.x[i][0], self.x[i][1], 1, 0, 0, 0])
                A[(2 * i) + 1] = np.array([0, 0, 0, self.x[i][0], self.x[i][1], 1])
                C[2 * i] = self.x_prime[i][0]
                C[(2 * i) + 1] = self.x_prime[i][1]
            Ainv = np.linalg.inv(A)
            B = np.dot(Ainv, C)
            B = np.append(B, (0,0,1))
            self.h = rearrange(B, '(c h)-> c h', c=self.h_mat_size, h=self.h_mat_size)
        return self.h


def harrisCornerDetector(img, filter="HAAR", sigma=1.2):
    # if filter=="SOBEL":
    #     gx, gy = Sobel()
    #
    # if filter=="HAAR":
    gx, gy = Haar_Wavelet(sigma)
    win_size = int(np.ceil(5 * sigma)) if np.ceil(5 * sigma) % 2 == 0 else int(np.ceil(5 * sigma)) + 1
    dx = convolve2d(img, gx, mode="same")
    dy = convolve2d(img, gy, mode="same")

    sumK = np.ones((win_size,win_size))
    dx2 = dx**2
    dy2 = dy**2
    dxdy= np.multiply(dx,dy)

    sum_dx2=convolve2d(dx2, sumK,mode="same")
    sum_dy2=convolve2d(dy2, sumK,mode="same")
    sum_dxdy=convolve2d(dxdy, sumK,mode="same")

    detC = sum_dx2*sum_dy2 - (sum_dxdy**2)
    trC = sum_dx2+sum_dy2
    trC2 = trC**2
    k=0.05
    Ratios = detC - k*trC2
    Ratios*(Ratios==maximum_filter(Ratios, footprint=np.ones((5,5))))
    corners = np.where(Ratios>0.01*np.max(Ratios))

    # print(len(corners[0]), len(corners[1]))
    return corners

def extarct_patch(pt, win_size, img):
    # print(img.shape)
    assert win_size%2 != 0
    pad = int((win_size-1)/2)
    val_minx = pt[0] - pad
    val_maxx = pt[0] + pad
    val_miny = pt[1] - pad
    val_maxy = pt[1] + pad
    win_minx = val_minx #if val_minx > 0 else 0
    win_maxx = val_maxx #if val_maxx < img.shape[1] else img.shape[1]
    win_miny = val_miny #if val_miny > 0 else 0
    win_maxy = val_maxy #if val_maxy < img.shape[0] else img.shape[0]
    # print(win_minx, win_miny, win_maxx, win_maxy)
    pix_vals = []
    for i in range(win_minx, win_maxx):
        for j in range(win_miny, win_maxy):
            pix_vals.append(img[j,i])
    return pix_vals

def ssd(pts1, pts2, img1, img2, win_size):
    combined_pt_set=[]
    patches1=[]
    patches2=[]
    pad=int((win_size-1)/2)
    img1=cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2=cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))

    for i in range(len(patches1)):
        min_dist =[]
        for j in range(len(patches2)):
            min_dist.append(np.mean((np.array(patches1[i])-np.array(patches2[j]))**2))
        min_val = min(v for v in min_dist if v>0.0)
        # combined_pt_set.append([i,np.argmin(np.array(min_dist)), min_dist[np.argmin(np.array(min_dist))]])
        combined_pt_set.append([i,min_dist.index(min_val), min_dist[min_dist.index(min_val)]])

    return combined_pt_set


def ncc(pts1, pts2, img1, img2, win_size):
    combined_pt_set = []
    patches1 = []
    patches2 = []
    pad = int((win_size - 1) / 2)
    img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    for pt in pts1:
        patches1.append(extarct_patch(pt, win_size, img1))
    for pt in pts2:
        patches2.append(extarct_patch(pt, win_size, img2))

    for i in range(len(patches1)):
        min_dist = []
        p1=np.array(patches1[i])
        for j in range(len(patches2)):
            p2 = np.array(patches2[j])
            numerator = np.sum((p1-np.mean(p1))*(p2-np.mean(p2)))
            denomiator= np.sqrt(np.sum((p1-np.mean(p1))**2)*np.sum((p2-np.mean(p2))**2))
            min_dist.append(1-(numerator/(denomiator+1e-6)))
        min_val = min(v for v in min_dist if v > 0.0)
        combined_pt_set.append([i,min_dist.index(min_val), min_dist[min_dist.index(min_val)]])

    return combined_pt_set

'''
Sift code from opencv documentation page.
https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
'''
def sift_matching(img_path1, img_path2):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp1, dv1 = sift.detectAndCompute(img1, None)
    kp2, dv2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(dv1,dv2,k=2)

    points = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            points.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3, points
def Sobel():
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return gx, gy

def Haar_Wavelet(sigma):
    size = int(np.ceil(4 * sigma)) if np.ceil(4 * sigma) % 2 == 0 else int(np.ceil(4 * sigma)) + 1
    gx = np.ones((size, size))
    gx = gx.astype(int)
    gx[:, :int(size / 2)] = -1
    gy = gx.T
    return gx, gy

def pick_color():
    return (random.randint(10,250),random.randint(10,250),random.randint(10,250))
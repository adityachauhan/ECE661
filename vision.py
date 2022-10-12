import os
from scipy.optimize import least_squares

import cv2
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm

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
    X_prime = X_prime.astype(int)
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
    if filter=="SOBEL":
        gx, gy = Sobel()

    if filter=="HAAR":
        gx, gy = Haar_Wavelet(sigma)

    win_size = int(np.ceil(5 * sigma)) if np.ceil(5 * sigma) % 2 == 0 else int(np.ceil(5 * sigma)) + 1
    dx = cv2.filter2D(img, -1, gx)
    dy = cv2.filter2D(img, -1, gy)
    temp = np.zeros((img.shape[0], img.shape[1]))
    padding = int((win_size)/2)
    new_dx = np.zeros(((img.shape[0]+2*padding),(img.shape[1]+2*padding)))
    new_dy = np.zeros(((img.shape[0]+2*padding),(img.shape[1]+2*padding)))
    new_dx[padding:padding+dx.shape[0], padding:padding+dx.shape[1]]=dx
    new_dy[padding:padding+dy.shape[0], padding:padding+dy.shape[1]]=dy
    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            sum_dx = new_dx[(i+padding):i+(2*padding)+1,(j+padding):j+(2*padding)+1]
            sum_dy = new_dy[(i+padding):i+(2*padding)+1,(j+padding):j+(2*padding)+1]
            sum_dx2 = np.sum(np.multiply(sum_dx,sum_dx))
            sum_dy2 = np.sum(np.multiply(sum_dy,sum_dy))
            sum_dxdy= np.sum(np.multiply(sum_dx,sum_dy))
            detC = sum_dx2*sum_dy2-(sum_dxdy**2)
            trC = (sum_dx2+sum_dy2)**2
            val=detC/(trC+1e-6)
            if val>0:temp[i,j]=val

    thresh = np.mean(temp)
    thresh = 0.1
    print(thresh)
    pts=[]
    nms_win_size=int(img.shape[0]/4)
    print(nms_win_size)
    padding = int((nms_win_size-1)/2)
    padded_temp = np.zeros(((img.shape[0]+2*padding),(img.shape[1]+2*padding)))
    padded_temp[padding:padding+temp.shape[0], padding:padding+temp.shape[1]]=temp
    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            if temp[i,j]>0:
                if temp[i,j]==np.max(padded_temp[(i+padding):i+(2*padding)+1,(j+padding):j+(2*padding)+1]) and temp[i,j]>thresh:
                    pts.append([j,i])

    return pts

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

def SIFTpoints(img):
    pts_set=[]
    sift = cv2.SIFT_create()
    pts, des = sift.detectAndCompute(img, None)

    for pt in pts:
        pts_set.append([int(pt.pt[0]), int(pt.pt[1])])
    return pts_set, des


def flann_matching(pts1,pts2,des1,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchingPts1=[]
    matchingPts2=[]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.25 * n.distance:
            matchingPts1.append((pts1[m.queryIdx].pt[0],pts1[m.queryIdx].pt[1]))
            matchingPts2.append((pts2[n.trainIdx].pt[0],pts2[n.trainIdx].pt[1]))
    matchingPts1 = [[int(val[0]), int(val[1])] for val in matchingPts1]
    matchingPts2 = [[int(val[0]), int(val[1])] for val in matchingPts2]
    return matchingPts1, matchingPts2

def hmat_pinv(x, x_prime):
    A = np.ones((len(x) * 2, 8))
    C = np.ones(len(x) * 2)
    # print(A.shape, C.shape)
    for i in range(len(x)):
        A[2 * i] = np.array([x[i][0], x[i][1], 1, 0, 0, 0, -x[i][0] * x_prime[i][0],
                             -x[i][1] * x_prime[i][0]])
        A[(2 * i) + 1] = np.array([0, 0, 0, x[i][0], x[i][1], 1, -x[i][0] * x_prime[i][1],
                                   -x[i][1] * x_prime[i][1]])
        C[2 * i] = x_prime[i][0]
        C[(2 * i) + 1] = x_prime[i][1]

    h = (np.linalg.inv((A.T)@A)@A.T)@C
    h = np.concatenate([h,[1]])
    h = rearrange(h, '(c h)-> c h',c=3, h=3)
    return h


# def RANSAC(pts1, pts2):
#     p = 0.99
#     n = 20
#     sigma=2
#     delta = 3*sigma
#     e = 0.1
#     N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
#     print(N)
#     n_total = len(pts1)
#     pts1_hc = np.ones((n_total, 3))
#     pts1_hc[:, :-1] = pts1
#     pts2_hc = np.ones((n_total, 3))
#     pts2_hc[:, :-1] = pts2
#     print(pts1_hc.shape, pts2_hc.shape)
#     outliers=True
#     while outliers:
#         count=0
#         prev_size=0
#         idx_set=[]
#         while count < N:
#             randIdx = np.random.randint(n_total, size=n)
#             cpts1 = pts1_hc[randIdx,:]
#             cpts2 = pts2_hc[randIdx,:]
#             cH = hmat_pinv(cpts1, cpts2)
#             est_cpts2 = cH @ pts1_hc.T
#             est_cpts2 = est_cpts2/est_cpts2[2]
#             est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
#             error = (pts2_hc[:,:-1] - est_cpts2[:,:-1])**2
#             error = error @ np.ones((2,1))
#             in_indices = np.where(error <= delta)[0]
#             if len(in_indices) > prev_size:
#                 idx_set = in_indices
#                 prev_size=len(in_indices)
#             count+=1
#         if len(idx_set) > (1-e)*n_total:
#             outliers=False
#             print("Here")
#         else:
#             e*=2
#             N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
#             print("new N", N)
#     print("num inliers found:", len(idx_set))
#     return idx_set

def RANSAC(pts1, pts2):
    p = 0.99
    n = 20
    sigma=4
    delta = 3*sigma
    e = 0.1
    N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
    print(N)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    # print(pts1_hc.shape, pts2_hc.shape)
    outliers=True
    while outliers:
        count=0
        prev_size=0
        idx_set=[]
        while count < N:
            randIdx = np.random.randint(n_total, size=n)
            cpts1 = pts1_hc[randIdx,:]
            cpts2 = pts2_hc[randIdx,:]
            cH = hmat_pinv(cpts1, cpts2)
            est_cpts2 = cH @ pts1_hc.T
            est_cpts2 = est_cpts2/est_cpts2[2]
            est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
            error = (pts2_hc[:,:-1] - est_cpts2[:,:-1])**2
            error = error @ np.ones((2,1))
            in_indices = np.where(error <= delta)[0]
            if len(in_indices) > prev_size:
                print("reached", len(in_indices), (1-e)*n_total)
                idx_set = in_indices
                if len(idx_set) > (1 - e) * n_total:
                    outliers = False
                    break
                prev_size=len(in_indices)
            count+=1
        if len(idx_set) > (1-e)*n_total:
            outliers=False
            print("Here")
        else:
            e*=2
            N = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** n)))
            print("new N", N)
    print("num inliers found:", len(idx_set))
    return idx_set


def sift_match(pts1, des1, pts2, des2, tr=3):
    match_pts1 = []
    match_pts2 = []
    eu = np.zeros((len(pts1), len(pts2)))
    for i in trange(len(pts1)):
        for j in range(len(pts2)):
            eu[i, j] = np.linalg.norm(des1[i, :] - des2[j, :])

    eu = eu / np.min(eu)
    dy_threshold = tr
    for i in trange(len(pts1)):
        eu_min = np.min(eu[i, :])
        if eu_min < dy_threshold:
            j_min = np.argmin(eu[i, :])
            match_pts1.append(pts1[i])
            match_pts2.append(pts2[j_min])
    print('number of sift matches found:', len(match_pts1))
    return match_pts1, match_pts2

def costFun(h, pts1, pts2):
    h = rearrange(h, '(c h)-> c h', c=3,h=3)
    n_total = len(pts1)
    pts1_hc = np.ones((n_total, 3))
    pts1_hc[:, :-1] = pts1
    pts2_hc = np.ones((n_total, 3))
    pts2_hc[:, :-1] = pts2
    est_cpts2 = h @ pts1_hc.T
    est_cpts2 = est_cpts2 / est_cpts2[2]
    est_cpts2 = rearrange(est_cpts2, 'c h -> h c')
    X = rearrange(np.array(pts2), 'c h -> (c h)')
    f = rearrange(np.array(est_cpts2[:,:-1]), 'c h -> (c h)')
    error = X-f
    # error = error @ np.ones((2,1))
    # error = rearrange(error, 'c h -> (c h)')
    return error


def pan(comb, img, H):
    h=img.shape[0]
    w=img.shape[1]
    h_tot=comb.shape[0]
    w_tot=comb.shape[1]
    H = np.linalg.inv(H)
    X,Y=np.meshgrid(np.arange(0,w_tot,1),np.arange(0,h_tot,1))
    pts=np.vstack((X.ravel(), Y.ravel())).T
    pts=np.hstack((pts[:,0:2],pts[:,0:1]*0+1))
    Hpts=H @ pts.T
    Hpts=Hpts/Hpts[2,:]
    Hpts=Hpts.T[:, 0:2].astype('int')
    valid_pts, valid_Hpts = find_valid_pts(pts, Hpts, w-1, h-1)
    for i in range(valid_pts.shape[0]):
        if (comb[valid_pts[i,1], valid_pts[i,0]] != 0).all() == False:
            comb[valid_pts[i, 1], valid_pts[i, 0]]=img[valid_Hpts[i,1], valid_Hpts[i,0]]
    return comb
def find_valid_pts(pts, Hpts, w,h):
    xmin=Hpts[:,0] >=0
    Hpts=Hpts[xmin,:]
    pts=pts[xmin,:]
    xmax=Hpts[:,0]<=w
    Hpts=Hpts[xmax,:]
    pts=pts[xmax,:]
    ymin=Hpts[:,1]>=0
    Hpts=Hpts[ymin,:]
    pts=pts[ymin,:]
    ymax=Hpts[:,1]<=h
    Hpts=Hpts[ymax,:]
    pts=pts[ymax,:]
    return pts, Hpts

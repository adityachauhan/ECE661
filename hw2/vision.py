import cv2
import numpy as np
from einops import rearrange

def readImgCV(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cvrt2homo(pt):
    return np.append(pt, 1)

class Vision:
    def __init__(self, x, x_prime, homo_mat_size=3):
        self.x = x
        self.x_prime = x_prime
        self.h = np.ones((homo_mat_size,homo_mat_size))
        self.h_mat_size = homo_mat_size

    def calc_homograpy(self):
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
        return self.h
<<<<<<< HEAD
=======





>>>>>>> 84adec30d19eacb1564ca1e85790bd68ed6e1587

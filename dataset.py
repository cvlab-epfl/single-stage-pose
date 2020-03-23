import numpy as np
from torch.utils.data import Dataset
import torch
import math
import random
import os

def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q

def quaternion2rotation(quat):
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)

class PnP_Data_Simulator(Dataset):
    def __init__(self, sampleCnt=20000, gridCnt=200, minNoiseSigma=0, maxNoiseSigma=0, minOutlier=0, maxOutlier=0):
        self.width = 640
        self.height = 480
        self.intrinsic = torch.from_numpy(np.array([[800, 0, self.width/2],
                                                    [0, 800, self.height/2],
                                                    [0, 0, 1]])).float()
        self.point_3d = 0.5 * torch.from_numpy(np.array([1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1])).float()
        self.point_3d = self.point_3d.view(-1, 3)

        self.gridCnt = gridCnt
        self.minNoiseSigma = minNoiseSigma
        self.maxNoiseSigma = maxNoiseSigma
        self.minOutlier = minOutlier
        self.maxOutlier = maxOutlier
        self.sampleCnt = sampleCnt
        # 
        self.translation_min = [-2,-2,4]
        self.translation_max = [2,2,8]

    def __len__(self):
        return self.sampleCnt

    def __getitem__(self, index):
        gt_r =self.RandomRotation()
        gt_q = torch.from_numpy(rotation2quaternion(gt_r)).float()
        gt_r = torch.from_numpy(gt_r).float()
        gt_t = torch.from_numpy(self.RandomTranslation()).float()

        # 
        # select grids randomly within the image plane
        sy = np.random.randint(self.height, size=self.gridCnt)
        sx = np.random.randint(self.width, size=self.gridCnt)
        sy = torch.from_numpy(sy.reshape(-1, 1).repeat(len(self.point_3d), axis=1)).float()
        sx = torch.from_numpy(sx.reshape(-1, 1).repeat(len(self.point_3d), axis=1)).float()
        # 
        # 2d reprojection
        p = torch.mm(self.intrinsic, torch.mm(gt_r, self.point_3d.t()) + gt_t.view(-1,1))
        tx = (p[0] / p[2]).view(1,-1)
        ty = (p[1] / p[2]).view(1,-1)
        dx = tx-sx
        dy = ty-sy
        
        sxy = torch.cat((sx.view(-1, 1), sy.view(-1, 1)), 1)
        dxy = torch.cat((dx.view(-1, 1), dy.view(-1, 1)), 1)

        # add outlier
        outlierRatio = np.random.uniform(self.minOutlier, self.maxOutlier)
        outlierCnt = int(len(dxy) * outlierRatio + 0.5)
        outlierChoice = np.random.choice(len(dxy), outlierCnt, replace=False)

        sxy[outlierChoice, 0] = torch.from_numpy(np.random.uniform(0, self.width-1, size=outlierCnt)).float()
        sxy[outlierChoice, 1] = torch.from_numpy(np.random.uniform(0, self.height-1, size=outlierCnt)).float()
        # 
        dxy[outlierChoice, 0] = torch.from_numpy(np.random.uniform(0, self.width-1, size=outlierCnt)).float()
        dxy[outlierChoice, 1] = torch.from_numpy(np.random.uniform(0, self.height-1, size=outlierCnt)).float()

        # add noise to 2d
        noiseSigma = np.random.uniform(self.minNoiseSigma, self.maxNoiseSigma)
        noise = np.random.normal(0, noiseSigma, (len(dxy), 2)).astype(np.float32)
        # 
        dxy = dxy + torch.from_numpy(noise)

        return self.intrinsic, gt_q, gt_t, sxy, dxy, self.point_3d

    def Rand(self, min, max):
        return min + (max - min) * random.random()

    def RandomRotation(self):
        range = 1

        # use eular formulation, three different rotation angles on 3 axis
        phi = self.Rand(0, range * math.pi * 2)
        theta = self.Rand(0, range * math.pi)
        psi = self.Rand(0, range * math.pi * 2)

        R0 = []
        R0.append(math.cos(psi) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.sin(psi))
        R0.append(math.cos(psi) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.sin(psi))
        R0.append(math.sin(psi) * math.sin(theta))

        R1 = []
        R1.append(-math.sin(psi) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.cos(psi))
        R1.append(-math.sin(psi) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.cos(psi))
        R1.append(math.cos(psi) * math.sin(theta))

        R2 = []
        R2.append(math.sin(theta) * math.sin(phi))
        R2.append(-math.sin(theta) * math.cos(phi))
        R2.append(math.cos(theta))

        R = []
        R.append(R0)
        R.append(R1)
        R.append(R2)
        return np.array(R)

    def RandomTranslation(self):
        tx = self.Rand(self.translation_min[0], self.translation_max[0])
        ty = self.Rand(self.translation_min[1], self.translation_max[1])
        tz = self.Rand(self.translation_min[2], self.translation_max[2])
        return np.array([tx, ty, tz]).reshape(-1)



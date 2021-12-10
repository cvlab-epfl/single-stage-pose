#%%
import tqdm 
from dataset import PnP_Data_Simulator
import torch
import cv2
import os
import numpy as np 
from dataset import *
from model import *
from train import quaternion2rotation

def pose_err_3d(kpts3d, r, t, gr, gt):
    t = t.reshape(-1, 1)
    gt = gt.reshape(-1, 1)
    # error metric in 3D space
    res1 = np.matmul(r, kpts3d.T) + t
    res2 = np.matmul(gr, kpts3d.T)+ gt
    diff = np.linalg.norm(res1-res2, axis=0)
    return diff.mean()

def pose_err_2d(kpts3d, r, t, gr, gt, K):
    t = t.reshape(-1, 1)
    gt = gt.reshape(-1, 1)
    # error metric in 3D space
    res1 = np.matmul(K, np.matmul(r, kpts3d.T) + t)
    x1 = res1[0]/res1[2]
    y1 = res1[1]/res1[2]
    res2 = np.matmul(K, np.matmul(gr, kpts3d.T)+ gt)
    x2 = res2[0]/res2[2]
    y2 = res2[1]/res2[2]
    diff = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return diff.mean()

def PnP_OpenCV_EPNP(p3d, p2d, intrinsics):
    ptCnt = len(p3d)
    retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
    # retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_P3P)

    if not retval:
        print('PnP Failed')
        R = np.eye(3)
        T = np.array([0,0,1]).reshape(-1, 1)
        return R, T
    R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
    T = trans.reshape(-1, 1)
    return R, T

def PnP_Learning(p3d, sxy, dxy, intrinsics, width, height, translation_min, translation_max, model=None):
    p3d = torch.from_numpy(p3d).float().unsqueeze(dim=0)
    sxy = torch.from_numpy(sxy).float().unsqueeze(dim=0)
    dxy = torch.from_numpy(dxy).float().unsqueeze(dim=0)

    sxy[..., 0] = sxy[..., 0] - (width/2)
    sxy[..., 1] = sxy[..., 1] - (height/2)
    sxy[..., 0] = sxy[..., 0] / width
    sxy[..., 1] = sxy[..., 1] / height

    # dxy = dxy / ( 2* dxy.norm(dim=-1).unsqueeze(-1) )
    dxy[..., 0] = dxy[..., 0] / width
    dxy[..., 1] = dxy[..., 1] / height

    inData = torch.cat((sxy, dxy), dim=-1)
    # inData = torch.cat((p3d, inData), dim=-1)
    inData = inData.transpose(1,2)

    out = model(inData)
    predQ = out[:, :4]
    predT = out[:, 4:]
    predR = quaternion2rotation(predQ)

    # recover predicted translation
    minT = translation_min.type_as(out).view(-1,3)
    maxT = translation_max.type_as(out).view(-1,3)
    predT =  (predT + 0.5) * (maxT - minT) + minT

    return predR[0].detach().numpy(), predT[0].detach().numpy().reshape(-1,1)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    batch_size = 32
    model = SimplePnPNet(nIn=4)
    model.eval()
    model.load_state_dict(torch.load('./xydxdy.pth'))

    min_noise = 0
    max_noise = 31
    print("Noise\tEPnP(3d)\tOurs(3d)\tEPnP(2d)\tOurs(2d)")
    # 
    for noise in range(min_noise, max_noise):
        dataset = PnP_Data_Simulator(sampleCnt=2000, minNoiseSigma=noise, maxNoiseSigma=noise, minOutlier=0.1, maxOutlier=0.1)
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

        translation_min = torch.FloatTensor(dataset.translation_min)
        translation_max = torch.FloatTensor(dataset.translation_max)

        # pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))  # progress bar
        err_3d_stat_opencv_epnp = []
        err_3d_stat_opencv_ours = []
        # 
        err_2d_stat_opencv_epnp = []
        err_2d_stat_opencv_ours = []

        # for batch_idx, batch_data in pbar:
        for batch_idx, batch_data in enumerate(data_loader):  
            intrinsic, quat, trans, sxy, dxy, p3ds = batch_data
            p3ds = p3ds.repeat(1,dataset.gridCnt,1)
            
            Rs = quaternion2rotation(quat)
            Ts = trans

            count = len(Rs)
            for i in range(count):
                k = intrinsic[i].numpy()
                gtR = Rs[i].numpy()
                gtT = Ts[i].numpy().reshape(-1, 1)
                p3d = p3ds[i].numpy()
                p2d = (sxy + dxy)[i].numpy()

                ptCnt = len(p3d)

                R1, T1 = PnP_OpenCV_EPNP(p3d, p2d, k)
                err = pose_err_3d(p3ds[i].numpy(), R1, T1, gtR, gtT)
                err_3d_stat_opencv_epnp.append(err)
                err = pose_err_2d(p3ds[i].numpy(), R1, T1, gtR, gtT, k)
                err_2d_stat_opencv_epnp.append(err)

                R2, T2 = PnP_Learning(p3ds[i].numpy(), sxy[i].numpy(), dxy[i].numpy(), k, dataset.width, dataset.height, \
                    translation_min, translation_max, model=model)
                err = pose_err_3d(p3ds[i].numpy(), R2, T2, gtR, gtT)
                err_3d_stat_opencv_ours.append(err)
                err = pose_err_2d(p3ds[i].numpy(), R2, T2, gtR, gtT, k)
                err_2d_stat_opencv_ours.append(err)

        print("%d\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f" % (noise, np.array(err_3d_stat_opencv_epnp).mean(), np.array(err_3d_stat_opencv_ours).mean(), np.array(err_2d_stat_opencv_epnp).mean(), np.array(err_2d_stat_opencv_ours).mean()))



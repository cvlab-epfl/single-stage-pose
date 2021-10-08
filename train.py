#%%
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
from torch.utils.data import Dataset

from dataset import *
from model import *

import os

from tensorboardX import SummaryWriter

def quaternion2rotation(quat):
    assert (quat.shape[1] == 4)
    # normalize first
    quat = quat / quat.norm(p=2, dim=1).view(-1, 1)

    a = quat[:, 0]
    b = quat[:, 1]
    c = quat[:, 2]
    d = quat[:, 3]

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

    return torch.stack((m0, m1, m2, m3, m4, m5, m6, m7, m8), dim=1).view(-1, 3, 3)

def compute_loss(pt_3d, predQ, predT, gtQ, gtT):
    q1 = predQ
    t1 = predT
    q2 = gtQ
    t2 = gtT
    r1 = quaternion2rotation(q1)
    r2 = quaternion2rotation(q2)
    # 
    # compute error in 3D
    res1 = torch.bmm(r1, pt_3d.transpose(1, 2)) + t1.unsqueeze(dim=2)
    res2 = torch.bmm(r2, pt_3d.transpose(1, 2)) + t2.unsqueeze(dim=2)

    diff = (res1-res2).norm(dim=1).mean(dim=1)
    return diff.mean()

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tag = 'xydxdy'

    logger = SummaryWriter('logs_' + tag)

    print('start training ...')
    model = SimplePnPNet(nIn=4)
    model = model.cuda()

    desired_epoch = 200
    batch_size = 32
    learning_rate = 1e-4
    alpha = 1
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(desired_epoch * x) for x in [0.5, 0.8, 0.9]], gamma=0.1)
    # 
    dataset = PnP_Data_Simulator(sampleCnt=20000, minNoiseSigma=0, maxNoiseSigma=15, minOutlier=0, maxOutlier=0.3)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    # 
    translation_min = torch.FloatTensor(dataset.translation_min)
    translation_max = torch.FloatTensor(dataset.translation_max)
    # 
    model.train()
    for epoch in range(desired_epoch):
        # Update scheduler
        if epoch > 0:
            scheduler.step()
        for batch_idx, batch_data in enumerate(data_loader):
            intrinsic, quat, trans, sxy, dxy, p3d = batch_data
            intrinsic = intrinsic.cuda()
            quat = quat.cuda()
            trans = trans.cuda()
            sxy = sxy.cuda()
            dxy = dxy.cuda()
            p3d = p3d.cuda()

            # normalize according to width and height
            # xy = sxy+dxy
            # normalize xy to [-0.5, 0.5]
            xy = sxy
            xy[..., 0] = xy[..., 0] - (dataset.width/2)
            xy[..., 0] = xy[..., 0] / dataset.width
            xy[..., 1] = xy[..., 1] - (dataset.height/2)
            xy[..., 1] = xy[..., 1] / dataset.height

            # normalize dxdy to [-0.5, 0.5]
            # dxy = dxy / ( 2* dxy.norm(dim=-1).unsqueeze(-1))
            dxy[..., 0] = dxy[..., 0] / dataset.width
            dxy[..., 1] = dxy[..., 1] / dataset.height
            # 
            # theta = torch.atan2(dxy[..., 1], dxy[..., 0])
            # theta = theta.unsqueeze(-1) / math.pi
            # inData = torch.cat((xy, theta), dim=-1)
            
            inData = torch.cat((xy, dxy), dim=-1)
            # inData = torch.cat((p3d.repeat(1,dataset.gridCnt,1), inData), dim=-1)
            # 
            inData = inData.transpose(1,2)

            # inData = expandFeatures(inData,deg=2)
            # target = torch.cat((gt_q,gt_t), dim=1)
            # predQ = model(inData)
            # loss = compute_loss(pt_3d, predQ, gt_t, gt_q, gt_t)
            # predT = model(inData)
            # loss = compute_loss(pt_3d, gt_q, predT, gt_q, gt_t)
            out = model(inData)
            predQ = out[:, :4]
            predT = out[:, 4:]

            # recover predicted translation
            minT = translation_min.type_as(out).view(-1,3)
            maxT = translation_max.type_as(out).view(-1,3)
            predT =  (predT + 0.5) * (maxT - minT) + minT

            loss = alpha * compute_loss(p3d, predQ, predT, quat, trans)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logger.add_scalar('loss', loss, epoch*len(data_loader) + batch_idx)

            if batch_idx % 10 == 0:
                print('epoch %d/%d, batch %d/%d, lr %f, %f' % (epoch, desired_epoch, batch_idx, len(data_loader), scheduler.get_lr()[0], loss))
        # 
        torch.save(model.state_dict(), tag + '.pth')

if __name__ == "__main__":
    train()

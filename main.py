import os
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
from manopth.manolayer import ManoLayer
from utils.joint2quat import get_quat_from_joints
from utils.losses import *
from utils.vis import vis_fit_process
from data.RHD.RHDdata import RHDdataset
from torch.utils.data import DataLoader
from utils.nmr import NMR
from utils.geometric import orthographic_proj_withz as project
from config import cfg

mano_layer = ManoLayer(center_idx = 0, use_pca = True, ncomps = cfg.ncomps, flat_hand_mean = True).cuda()

db = RHDdataset(root=os.path.join(cfg.data_loc, cfg.dataset), mode='train')
db_loader = DataLoader(db, batch_size = 1, shuffle = False)

class MANOFIT(torch.nn.Module):
    def __init__(self, theta):
        super(MANOFIT, self).__init__()
        self.cam = torch.nn.Parameter(theta[:,0:4])
        self.pose_param = torch.nn.Parameter(theta[:, 4:52])
        self.shape_param = torch.nn.Parameter(theta[:, 52:])
    def forward(self):
        verts, joints, faces = mano_layer(self.pose_param, self.shape_param)
        joints2D = project(joints, cam= self.cam) #project verts too
        joints2D[:,:,:] = joints2D[:,:,:]*0.5 +0.5
        verts = project(verts, cam=self.cam)
        return verts, joints, joints2D, faces


mask_renderer = NMR(image_size = 224).cuda()
joint_loss = torch.nn.MSELoss()

for i, sample in enumerate(db_loader):
    print("img #: ",i)
    depths_GT, joints_GT, masks_GT  = sample['depths'], sample['joints'], sample['masks']
    quats = get_quat_from_joints(joints_GT)
    pose_param = mano_layer.pose_quat2pca(quats).cuda()
    shape_param = 0.001*torch.ones(1, 10).cuda()

    theta = 8*torch.ones([1, 1]).float().cuda() #cam
    theta = torch.cat((theta, torch.zeros([1, 3]).float().cuda()), dim =1) #trans
    theta = torch.cat((theta, pose_param, shape_param), dim = 1) #param

    model = MANOFIT(theta).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for l in range(cfg.fitting_iter_max):
        print("iter #: ", l)
        verts_pred, joints3D_pred, joints2D_pred, faces = model.forward()
        laplacian_loss = LaplacianLoss(faces.reshape([1, 1538, 3]))
        sil, dep= mask_renderer(verts_pred, faces, part=None, depth=True)
        if l % cfg.vis_interval == 0 :
                vis_fit_process(i, l, masks_GT, joints_GT, sil, joints2D_pred, dep)
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()

            if l < cfg.fitting_iter_max:
                #global fitting
                loss = 0.05*reproj_iou_loss(masks_GT.float().cuda(), sil) + angle_loss_3D(joints_GT.float().cuda(), joints2D_pred) + joint_loss(joints_GT.float().cuda(), joints2D_pred) + laplacian_loss(verts_pred)
                loss.backward()
                optimizer.step()
            else:
                break
                #local fitting, not complete
                # not_thumb = np.arange(3,39) #[1,2,3,4,5,6,7,8,9,10,11,12] # 0 13 14 15
                # not_index = np.arange(12,48) #[4,5,6,7,8,9,10,11,12,13,14,15] # 0 1 2 3
                # not_middle = np.concatenate([np.arange(3,12), np.arange(21, 48)], axis=0)#[1,2,3],[7,8,9,10,11,12,13,14,15] # 0 4 5 6
                # not_ring = np.concatenate([np.arange(3,30), np.arange(39,48)], axis=0)#[1,2,3,4,5,6,7,8,9],[13,14,15] #0 10 11 12
                # not_little= np.concatenate([np.arange(3,21), np.arange(30, 48)], axis=0)#[1,2,3,4,5,6],[10,11,12,13,14,15] # 0 7 8 9
                #
                # ori_cam = model.cam.clone().detach()
                # ori_pose = model.pose_param.clone().detach()
                #
                # if l % 5== 0:
                #     loss = finger_loss_3D(joints_GT.float().cuda(), joints2D_pred, finger= 'thumb') + 0.05*reproj_iou_loss(masks_GT.float().cuda(), sil)
                #     loss.backward(retain_graph=True)
                #     optimizer.step()
                #     with torch.no_grad():
                #         model.pose_param[:,not_thumb] = ori_pose[:,not_thumb]
                #         model.cam[:,:] = ori_cam[:,:]
                # elif l % 5 == 1:
                #     loss = finger_loss_3D(joints_GT.float().cuda(), joints2D_pred, finger='index') + 0.05*reproj_iou_loss(masks_GT.float().cuda(), sil)
                #     loss.backward(retain_graph=True)
                #     optimizer.step()
                #     with torch.no_grad():
                #         model.pose_param[:, not_index] = ori_pose[:, not_index]
                #         model.cam[:, :] = ori_cam[:, :]
                # elif l % 5 == 2:
                #     loss = finger_loss_3D(joints_GT.float().cuda(), joints2D_pred, finger='middle') + 0.05*reproj_iou_loss(masks_GT.float().cuda(), sil)
                #     loss.backward(retain_graph=True)
                #     optimizer.step()
                #     with torch.no_grad():
                #         model.pose_param[:, not_middle] = ori_pose[:, not_middle]
                #         model.cam[:, :] = ori_cam[:, :]
                # elif l % 5 ==3:
                #     loss = finger_loss_3D(joints_GT.float().cuda(), joints2D_pred, finger='ring') + 0.05*reproj_iou_loss(masks_GT.float().cuda(), sil)
                #     loss.backward(retain_graph=True)
                #     optimizer.step()
                #     with torch.no_grad():
                #         model.pose_param[:, not_ring] = ori_pose[:, not_ring]
                #         model.cam[:, :] = ori_cam[:, :]
                # elif l % 5 == 4:
                #     loss = finger_loss_3D(joints_GT.float().cuda(), joints2D_pred, finger='little') + 0.05*reproj_iou_loss(masks_GT.float().cuda(), sil)
                #     loss.backward(retain_graph=True)
                #     optimizer.step()
                #     with torch.no_grad():
                #         model.pose_param[:, not_little] = ori_pose[:, not_little]
                #         model.cam[:, :] = ori_cam[:, :]

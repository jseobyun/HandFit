import scipy.io as sio
import os
import numpy as np
import struct
from torch.utils.data import Dataset

'''
joint min Depth = 349.96
joint max Depth = 1078.65
joint avg Depth = 701.48

num_joints = 14
size_trainset = 72757
im_w = 640
im_h = 480
fx = 588.036865
fy = 587.075073
'''


def pixel2world(x, y, z, im_h=480 , im_w= 640, fx=588.036865, fy=587.075073):
    x_world = (x - im_w/2)* z / fx
    y_world = (y - im_h/2)* z / fy
    z_world = z
    return x_world, y_world, z_world

def world2pixel(x, y, z, im_h=480, im_w=640, fx=588.036865, fy=587.075073):
    x_pixel = fx*x/z+im_w/2
    y_pixel = im_h/2 -fy*y/z
    return x_pixel, y_pixel

def depthmap2points(depth_img):
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w)+1 , np.arange(h)+1)
    points = np.zeros([h,w,3], dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x,y,depth_img)
    return points

def load_depthmap(filename, im_h=480, im_w=640, max_depth= 1200):
    with open(filename, mode='rb') as f:
        import torch
        data = f.read()
        depthmap = struct.unpack('f'*480*640, data)
        depthmap = np.reshape(depthmap, (im_h, im_w))
        depthmap[depthmap==0] = max_depth
        return depthmap

class NYUdataset(Dataset):
    def __init__(self, root, mode, kinect_idx, transform=None):
        self.im_h = 480
        self.im_w = 640
        self.min_depth = 349.96
        self.max_depth = 1078.65
        self.fx = 588.036865
        self.fy = 587.075073
        self.num_joints = 14
        self.world_dim = 3
        self.kinect_idx = kinect_idx
        self.num_trainset = 72757
        self.num_testset = 8252
        self.eval_joint = [1, 4, 7, 10, 13, 16, 19, 22, 25, 26, 28, 31, 32, 33]

        self.root = root
        self.mode = mode
        self.transform = transform

        if not self.mode in ['train', 'test']:
            raise ValueError('Invalid mode')
        if not self._check_exists():
            raise RuntimeError('Invalid NYU dataset')
        self._load()

    def __getitem__(self, index):
        depthmap = load_depthmap(self.names[index])
        points = depthmap2points(depthmap)
        points = points.reshape((-1,3))
        sample = {
            'name' : self.names[index],
            'points' : points,
            'joints' : self.joints_world[index],
            'refpoint' : self.ref_pts[index]
        }
        if self.transform:
            sample = self.transform(sample)
        print("")
        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self.num_samples = self.num_trainset if self.mode == 'train' else self.num_testset
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))


        ref_pts_path = 'center/center_train_refined.txt' if self.mode =='train' else 'center/center_test_refined.txt'
        ref_pts_path = os.path.join(self.root, ref_pts_path)
        with open(ref_pts_path) as f:
            ref_pts_str = [l.rstrip() for l in f]

        joints_world_path = 'train_bin/joint_xyz.mat' if self.mode =='train' else 'test_bin/joint_xyz.mat'
        joints_world_path = os.path.join(self.root, joints_world_path)
        self.joints_world = sio.loadmat(joints_world_path)['joint_xyz'][self.kinect_idx]
        self.joints_world = self.joints_world[:,self.eval_joint,:]
        self.names = []

        for i in range(self.num_samples):
            splitted = ref_pts_str[i].split()
            if np.shape(splitted)[0] != self.world_dim:
                print('Warning: found invalid reference frame')
                continue
            else:
                self.ref_pts[i, 0] = float(splitted[0])
                self.ref_pts[i, 1] = float(splitted[1])
                self.ref_pts[i, 2] = float(splitted[2])
            filename = 'depth_'+ str(self.kinect_idx) +'_{:0>7d}'.format(i+1) + '.bin'
            if self.mode == 'train':
                filename = os.path.join(self.root, 'train_bin/depth'+str(self.kinect_idx), filename)
            else:
                filename = os.path.join(self.root, 'test_bin/depth' + str(self.kinect_idx), filename)
            self.names.append(filename)


    def _check_exists(self):
        if self.mode == 'train':
            joints_path = os.path.join(self.root, 'train_bin/joint_data.mat')
            center_path = os.path.join(self.root, 'center/center_train_refined.txt')
        else:
            joints_path = os.path.join(self.root, 'test_bin/joint_data.mat')
            center_path = os.path.join(self.root, 'center/center_train_refined.txt')
        if not os.path.exists(joints_path):
            print("Error : {} does not exist".format(joints_path))
            return False
        if not os.path.exists(center_path):
            print("Error : {} does not exist".format(center_path))

        return True














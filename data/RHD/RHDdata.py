import os
import numpy as np
from torch.utils.data import Dataset

def rearrange_joints(J):
    order = [0,4,3,2,1,8,7,6,5,12,11,10,9, 16,15,14,13, 20, 19, 18, 17]
    J = J[:,order,:]
    return J

class RHDdataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.im_h = 224
        self.im_w = 224
        self.num_joints = 21
        self.num_samples = 23899 if self.mode =='train' else 1359#41258 , 2728
        self.dir_path = os.path.join(root, "training") if self.mode == 'train' else os.path.join(root, "evaluation")

        if not self.mode in ['train', 'evaluation', 'test']:
            raise ValueError('Invalid mode')
        if not self._check_exists():
            raise RuntimeError('Invalid RHD dataset')

        self._load()

    def __getitem__(self, index):
        joints = self.joints[index]
        depths = self.depths[index]
        masks = self.masks[index]

        sample = {
            'depths' : depths,
            'joints' : joints,
            'masks' : masks
        }

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self.joints = np.load(
            os.path.join(self.dir_path, "total_train_joints_wrist.npy")) if self.mode == 'train' else np.load(
            os.path.join(self.dir_path, "total_eval_joints_wrist.npy"))

        self.masks = np.load(
            os.path.join(self.dir_path, "total_train_masks_wrist.npy")) if self.mode == 'train' else np.load(
            os.path.join(self.dir_path, "total_eval_masks_wrist.npy"))

        self.depths = np.load(
            os.path.join(self.dir_path, "total_train_images_wrist.npy")) if self.mode == 'train' else np.load(
            os.path.join(self.dir_path, "total_eval_images_wrist.npy"))

        self.joints = rearrange_joints(self.joints)


    def _check_exists(self):
        if self.mode == 'train':
            joint_path = os.path.join(self.dir_path, 'total_train_joints_wrist.npy')
            depth_path = os.path.join(self.dir_path, 'total_train_images_wrist.npy')
            mask_path = os.path.join(self.dir_path, 'total_train_masks_wrist.npy')
        else:
            joint_path = os.path.join(self.dir_path, 'total_eval_joints_wrist.npy')
            depth_path = os.path.join(self.dir_path, 'total_eval_images_wrist.npy')
            mask_path = os.path.join(self.dir_path, 'total_eval_masks_wrist.npy')

        if not os.path.exists(joint_path):
            print("Error : {} does not exist".format(joint_path))
            return False
        if not os.path.exists(mask_path):
            print("Error : {} does not exist".format(mask_path))
            return False
        if not os.path.exists(depth_path):
            print("Error : {} does not exist".format(depth_path))
            return False
        return True
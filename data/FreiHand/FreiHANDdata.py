import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset

class FreiHANDdataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.im_h = 224
        self.im_w = 224
        self.num_joints = 21
        self.num_samples = 32560 if self.mode =='train' else 3960
        self.dir_path = os.path.join(root, "training") if self.mode == 'train' else os.path.join(root, "evaluation")

        if not self.mode in ['train', 'evaluation', 'test']:
            raise ValueError('Invalid mode')
        if not self._check_exists():
            raise RuntimeError('Invalid FreiHAND dataset')

        self._load()



    def __getitem__(self, index):
        param = np.asarray(self.mano[index])
        scale = self.scale[index]
        joint = np.asarray(self.joints[index])
        K = np.asarray(self.K[index])

        name = "{0:08d}".format(index)+".jpg"
        rgb = cv2.imread(os.path.join(self.dir_path, "rgb/"+name))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        mask = False
        if self.mode == 'train':
            mask = cv2.imread(os.path.join(self.dir_path, "rgb/" + name))
            mask = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        sample = {
            'param': param,
            'joint': joint,
            'rgb': rgb,
            'mask': mask,
            'scale' : scale,
            'K' : K,
        }

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        if self.mode == 'train':
            self.mano = open(os.path.join(self.root, "training_mano.json")).read()
            self.mano = json.loads(self.mano)
            self.scale = open(os.path.join(self.root, "training_scale.json")).read()
            self.scale = json.loads(self.scale)
            self.joints = open(os.path.join(self.root, "training_xyz.json")).read()
            self.joints = json.loads(self.joints)
            self.K = open(os.path.join(self.root, "training_K.json")).read()
            self.K = json.loads(self.K)
        else:
            self.scale = open(os.path.join(self.root, "evaluation_scale.json")).read()
            self.scale = json.loads(self.scale)
            self.K = open(os.path.join(self.root, "evaluation_K.json")).read()
            self.K = json.loads(self.K)

    def _check_exists(self):
            if self.mode == 'train':
                mano_path = os.path.join(self.root, 'training_mano.json')
                joint_path = os.path.join(self.root, 'training_xyz.json')
                scale_path = os.path.join(self.root, 'training_scale.json')
                K_path = os.path.join(self.root, 'training_K.json')
            else:
                scale_path = os.path.join(self.root, 'evaluation_scale.json')
                K_path = os.path.join(self.root, 'evaluation_K.json')

            if self.mode == 'train':
                if not os.path.exists(joint_path) and not os.path.exists(mano_path) and not os.path.exists(scale_path) and not os.path.exists(K_path) :
                    print("Error : Train FreiHAND dataset does not exist")
                    return False
            else:
                if not os.path.exists(scale_path) and not os.path.exists(K_path):
                    print("Error : Eval FreiHAND dataset does not exist")
                    return False
            return True
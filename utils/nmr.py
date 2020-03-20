import numpy as np
import os
import torch
import torch.nn as nn
import neural_renderer as nr
from manopth.manolayer import ManoLayer
from config import cfg

class NMR(nn.Module):
    def __init__(self, image_size=320):
        super(NMR, self).__init__()
        self.renderer = nr.Renderer(image_size = image_size, camera_mode = 'look', perspective = False)
        self.image_size = image_size
        self.face_mask = np.load(os.path.join(cfg.root_dir, "face_mask.npy"), allow_pickle = True)
        self.manolayer = ManoLayer(use_pca = True, ncomps=45, flat_hand_mean = False)


    def forward(self, verts, faces, part=None, depth = False):
        batch_size = np.shape(verts)[0]

        verts[:,:,1] = -verts[:,:,1] # coordinate mismatch, invert y axis

        faces = faces.repeat(batch_size, 1,1).int()

        if part != None:
            fmask = self.face_mask[part]
            faces= faces[:,fmask,:]

        silhouettes = self.renderer(verts, faces, mode='silhouettes')
        if depth :
            depth_image = self.renderer(verts, faces, mode= 'depth')
            return silhouettes, depth_image
        else :
            return silhouettes
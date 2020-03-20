import numpy as np
import torch
from data.NYU.NYUdata import NYUdataset

def discretize(coord, cropped_size):
    min_normalized = -1
    max_normalized = 1
    scale = (max_normalized - min_normalized) / cropped_size
    return (coord -min_normalized ) / scale

def scattering(coord, cropped_size):
    coord = coord.astype(np.int32)
    mask = (coord[:, 0] >= 0) & (coord[:, 0] < cropped_size) & \
           (coord[:, 1] >= 0) & (coord[:, 1] < cropped_size) & \
           (coord[:, 2] >= 0) & (coord[:, 2] < cropped_size)
    coord = coord[mask, :]

    cubic = np.zeros((cropped_size, cropped_size, cropped_size))
    cubic[coord[:,0], coord[:,1], coord[:,2]] = 1
    return cubic


def generate_coord(points, refpoint, new_size, angle, trans, sizes):
    cubic_size, cropped_size, original_size = sizes

    coord = points
    coord = (coord - refpoint) / (cubic_size/2)
    print("x ", min(coord[:, 0]), max(coord[:,0]))
    print("y ", min(coord[:, 1]), max(coord[:,1]))
    print("z ", min(coord[:, 2]), max(coord[:,2]))

    coord = discretize(coord, cropped_size)
    print("x ", min(coord[:, 0]), max(coord[:, 0]))
    print("y ", min(coord[:, 1]), max(coord[:, 1]))
    print("z ", min(coord[:, 2]), max(coord[:, 2]))
    coord += (original_size /2 -cropped_size /2)

    resize_scale =new_size /100
    if new_size != 100:
        coord = coord *resize_scale + original_size/2 *(1-resize_scale)
    else:
        pass

    if angle !=0:
        original_coord = coord.copy()
        original_coord[:,0] -= original_size /2
        original_coord[:,1] -= original_size /2 # only xy plane rotation, move center
        coord[:,0] = original_coord[:,0]*np.cos(angle) - original_coord[:,1]*np.sin(angle)
        coord[:,1] = original_coord[:,0]*np.sin(angle) + original_coord[:,1]*np.cos(angle) #rotate
        coord[:,0] += original_size/2
        coord[:,1] += original_size/2 #move back

    coord -= trans
    return coord

def generate_cubic_input(points, refpoint, new_size, angle ,trans, sizes):
    _, cropped_size, _ = sizes
    coord = generate_coord(points, refpoint, new_size, angle, trans, sizes)
    cubic = scattering(coord, cropped_size)
    return cubic

def generate_joint_gt(joints, refpoint, new_size, angle, trans, sizes):
    _, cropped_size, _ = sizes
    joints_gt = generate_coord(joints, refpoint, new_size, angle, trans, sizes)
    return joints_gt

class Voxelization(object):
    def __init__(self, cubic_size, augmentation = False):
        self.cubic_size = cubic_size
        self.cropped_size, self.original_size = 88,96
        self.sizes = (self.cubic_size, self.cropped_size, self.original_size)
        self.augmentation = augmentation
        self.d3outputs = np.meshgrid(np.arange(self.cropped_size),np.arange(self.cropped_size),np.arange(self.cropped_size))

    def __call__(self, sample):
        name, points, joints, refpoint = sample['name'], sample['points'], sample['joints'], sample['refpoint']

        if self.augmentation:
            new_size = np.random.rand()*40 + 80 #80~120, 0.8~1.2
            angle = np.random.rand() * 80/180*np.pi - 40/180*np.pi # -40 ~ 40
            trans = np.random.rand(3)* (self.original_size - self.cropped_size)
        else :
            new_size = 100
            angle = 0
            trans = self.original_size/2 - self.cropped_size/2

        input = generate_cubic_input(points, refpoint, new_size, angle, trans, self.sizes)
        joints_gt = generate_joint_gt(joints, refpoint, new_size, angle ,trans, self.sizes)

        return input.reshape((1, *input.shape)), joints_gt

#######################################################################################################################
root = '/media/yjs/NYU/' # SSD path
mode = 'train'
kinect_idx = 1
num_joints = 14
batch_size = 1

vox_size = 88

voxelization_train = Voxelization(cubic_size = 300, augmentation= False)

def transform_train(sample):
    name, points, joints, refpoint = sample['name'], sample['points'], sample['joints'], sample['refpoint']
    assert (joints.shape[0] == num_joints)
    input, joints_gt= voxelization_train(sample)
    return (torch.from_numpy(input), torch.from_numpy(joints_gt))

dataset = NYUdataset(root = root, mode = mode, kinect_idx = 1, transform = transform_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)

for i, (inputs, joints_gts) in enumerate(dataloader):
    print("inputs shape", np.shape(inputs))
    print("joints gts ", np.shape(joints_gts))
    inputs = np.reshape(inputs.data.cpu().numpy(), [vox_size,vox_size,vox_size])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(inputs)
    plt.show()

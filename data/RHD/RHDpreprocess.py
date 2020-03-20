import pickle

import os
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageOps
from config import cfg
from mpl_toolkits.mplot3d import Axes3D

# chose between training and evaluation set
# set = 'training'
set = 'evaluation'
folder_path = os.path.join(cfg.data_loc, 'RHD', set)


# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2 ** 8 + bottom_bits).astype('float32')
    depth_map /= float(2 ** 16 - 1)
    depth_map *= 5.0
    return depth_map



# load annotations of this set
with open(os.path.join(folder_path, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

count = 1
bbsize = 400
center_joint = 0
# iterate samples of the set
total_images = []
total_masks = []
total_joints = []
for sample_id, anno in anno_all.items():
    count += 1
    kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
    if np.sum(kp_visible[21:]) != 21:
        continue
    print("sample_id", sample_id)
    kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
    camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters
    fx = camera_intrinsic_matrix[0, 0]
    fy = camera_intrinsic_matrix[1, 1]

    # load data
    image = scipy.misc.imread(os.path.join(folder_path, 'color', '%.5d.png' % sample_id))

    img = Image.open(os.path.join(folder_path, 'color', '%.5d.png' % sample_id))

    depth = scipy.misc.imread(os.path.join(folder_path, 'depth', '%.5d.png' % sample_id))
    depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])*1000  # depth in meters from the camera -> mm
    mask = scipy.misc.imread(os.path.join(folder_path, 'mask', '%.5d.png' % sample_id))
    mask = np.asarray(mask >= 18, dtype=np.int)
    gt = np.argwhere(mask ==1) # r,c
    gtu, gtv, gtd = gt[:,1], gt[:,0], depth[gt[:,0], gt[:,1]]
    ptu, ptv, ptd = kp_coord_uv[21:, 0], kp_coord_uv[21:,1], kp_coord_xyz[21:,2]*1000

    mean_u, mean_v, mean_z= ptu[center_joint], ptv[center_joint], ptd[center_joint]
    norm_size = bbsize / mean_z * fx

    # min_r = mean_v - 0.5 * norm_size
    # min_c = mean_u - 0.5 * norm_size
    # max_r = mean_v + 0.5 * norm_size
    # max_c = mean_u + 0.5 * norm_size
    #
    # img = ImageOps.expand(img, (200, 200, 200, 200))
    # print("min r, min c = {}, {}".format(mean_v - 0.5 * norm_size, mean_u - 0.5 * norm_size))
    # print("max r, max c = {}, {}".format(mean_v + 0.5 * norm_size, mean_u + 0.5 * norm_size))
    # cropped_image = img.crop((min_c+200, min_r+200, max_c+200, max_r+200))
    # cropped_image = cropped_image.resize((224,224))
    # cropped_image.save(("C:/Users/jongsoeb/Desktop/RHD_image/evaluation/" +  '%.5d.jpg' % sample_id))

    ptu = (ptu - mean_u) / norm_size + 0.5
    ptv = (ptv - mean_v) / norm_size + 0.5
    ptd = (ptd - mean_z) / bbsize + 0.5

    gtu = (gtu - mean_u) / norm_size + 0.5
    gtv = (gtv - mean_v) / norm_size + 0.5
    gtd = (gtd - mean_z) / bbsize + 0.5

    ptu = np.where(ptu > 1, 1, ptu)
    ptv = np.where(ptv > 1, 1, ptv)
    ptd = np.where(ptd > 1, 1, ptd)
    gtu = np.where(gtu > 1, 1, gtu)
    gtv = np.where(gtv > 1, 1, gtv)
    gtd = np.where(gtd > 1, 1, gtd)

    joints = np.asarray([ptu, ptv, ptd], dtype=np.float).transpose()

    newhand = -22*np.ones([int(norm_size), int(norm_size)], dtype= np.float)
    for idx in range(len(gtd)):
        ridx = int(gtv[idx]*norm_size)
        ridx = min(int(norm_size)-1, ridx)
        cidx = int(gtu[idx]*norm_size)
        cidx = min(int(norm_size) - 1, cidx)
        if newhand[ridx, cidx] == -22:
            newhand[ridx, cidx] = gtd[idx]
        elif newhand[ridx, cidx] > gtd[idx]:
            newhand[ridx, cidx] = gtd[idx]
    newhand[newhand == -22] = 0
    img_size = 224
    newhand = cv2.resize(newhand, dsize = (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # fig =plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.scatter(joints[:,0]*img_size, joints[:,1]*img_size, color= 'r')
    # ax1.imshow(newhand)
    # ax2 = fig.add_subplot(212)

    newhand_mask = newhand.copy()
    newhand_mask[newhand_mask !=0] = 1

    # ax2.imshow(newhand_mask)
    # plt.savefig(os.path.join("E:/yjs_data/RHD/train_images_wrist/", str(sample_id)+".png"))
    # plt.close()
    total_images.append(newhand)
    total_joints.append(joints)
    total_masks.append(newhand_mask)


print(np.shape(total_images))
print(np.shape(total_masks))
print(np.shape(total_joints))
np.save("E:/yjs_data/RHD/total_train_images_wrist.npy", total_images)
np.save("E:/yjs_data/RHD/total_train_masks_wrist.npy", total_masks)
np.save("E:/yjs_data/RHD/total_train_joints_wrist.npy", total_joints)


################################################
    # Project world coordinates into the camera frame
    # kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
    # kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]




# Visualize data
#     fig = plt.figure(1)
#     ax1 = fig.add_subplot('221')
#     ax2 = fig.add_subplot('222')
#     ax3 = fig.add_subplot('223')
#     ax4 = fig.add_subplot('224', projection='3d')
#
#     ax1.imshow(image)
#     ax1.plot(kp_coord_uv[33, 0], kp_coord_uv[33, 1], 'ro')
#     #         ax1.plot(kp_coord_uv_proj[kp_visible, 0], kp_coord_uv_proj[kp_visible, 1], 'gx')
#     ax2.imshow(depth)
#     ax3.imshow(mask)
#     ax4.scatter(kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
#     ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
#     ax4.set_xlabel('x')
#     ax4.set_ylabel('y')
#     ax4.set_zlabel('z')
#
#     plt.show()


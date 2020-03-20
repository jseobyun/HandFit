import numpy as np
import torch
from pyquaternion import * #Quaternion
def rearrange_joints(joints):
    rearrange_order= [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
    return joints[rearrange_order, :]

def normalize_vector(vecs): # vec : batch_size,
    norm = np.linalg.norm(vecs)
    if norm!= 0:
         vecs = vecs/norm
    return vecs

def get_quat_from_vecs(fromVector, toVector):
    axis= np.cross(fromVector, toVector)
    axis = normalize_vector(axis)

    fromVector= normalize_vector(fromVector)
    toVector= normalize_vector(toVector)
    angle = np.dot(fromVector, toVector) #0~pi
    angle = np.arccos(angle)
    quat = Quaternion(axis=axis, angle=angle)
    return quat, axis

def change_quats2array(quats):
    array = np.zeros([len(quats), 4])
    for jidx in range(len(quats)):
        w, x, y, z = quats[jidx] # batch_size, quat
        array[jidx, :] = [w,x,y,z]
    return array

def get_quat_from_joints(J_ori):

    child = [-1, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14,15, 16, -1, -1, -1, -1 -1]
    parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]

    J = J_ori[0].numpy()
    J = J.reshape(21, 3)
    J = rearrange_joints(J)

    quats = []

    target_m = J[4, :] - J[0, :]
    target_i = J[1, :] - J[0, :]
    target_ori = np.cross(target_m, target_i)

    MANO_m = [-94.6604, -1.4790, -3.3575]
    MANO_i = [-88.0973, -5.2004, 20.2860]
    MANO_ori = np.cross(MANO_m, MANO_i)
    global_rot1, _ = get_quat_from_vecs(MANO_ori, target_ori)

    rot_MANO_m = global_rot1.rotate(MANO_m)
    global_rot2, _ = get_quat_from_vecs(rot_MANO_m, target_m)
    global_rot = global_rot2*global_rot1
    inv_global_rot = global_rot.inverse

    quats.append(global_rot)

    for i in range(len(J)):
        J[i, :] = inv_global_rot.rotate(J[i, :])

    for i in range(1,16):
        vec_from_parent = J[i,:] - J[parent[i], :]
        vec_to_child = J[child[i], :] - J[i,:]
        quat, axis= get_quat_from_vecs(vec_from_parent, vec_to_child)
        quats.append(quat)

    quats_array = change_quats2array(quats).reshape(-1)
    quats_array = torch.from_numpy(quats_array).float().cuda()
    return quats_array.unsqueeze(dim=0)

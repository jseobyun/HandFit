import torch
import numpy as np

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, tz]
    Orth preserving the z.
    """

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X

    proj_xy = proj[:, :, :2] + trans[:, :, :2]
    proj_z = proj[:, :, 2] + trans[:, :, 2] + offset_z
    proj_z = proj_z[:, :, None]  # unsqueeze last dimension effect

    return torch.cat((proj_xy, proj_z), 2)

def normalize_tensor(vec):
    # B N 3

    norm = torch.norm(vec.clone(), dim=2)
    if np.shape(vec)[2] == 2:
        norm = torch.stack ((norm, norm), dim=2)
    else:
        norm = torch.stack((norm, norm, norm), dim=2)
    valid = norm !=0

    vec[valid] = vec[valid]/norm[valid]
    return vec

def dot_product(qa, qb):
    dot = torch.mul(qa, qb)
    dot = torch.sum(dot, dim=2)
    return dot

def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1 * qb_2 - qa_2 * qb_1
    q_mult_1 = qa_2 * qb_0 - qa_0 * qb_2
    q_mult_2 = qa_0 * qb_1 - qa_1 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]

    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]] * 0 + 1
    q = torch.unsqueeze(q, 1) * ones_x

    q_conj = torch.cat([q[:, :, [0]], -1 * q[:, :, 1:4]], dim=-1)
    X = torch.cat([X[:, :, [0]] * 0, X], dim=-1)

    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]
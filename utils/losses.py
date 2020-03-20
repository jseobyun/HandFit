import numpy as np
import torch
from utils.geometric import dot_product, cross_product, normalize_tensor
from scipy import sparse
def angle_loss_3D(j_gt_, j_pred_):
    #B, J ,3
    angle_list = [[0,1,2], [1,2,3], [2,3,4], [0,5,6], [5,6,7], [6,7,8], [0,9,10], [9,10,11], [10,11,12], [0,13,14], [13,14,15], [14,15,16], [0,17,18], [17,18,19], [18,19,20]]
    j_pred = j_pred_[:, angle_list]
    j_gt = j_gt_[:, angle_list]
    parent_pred = j_pred[:,:,2,:] - j_pred[:,:,1,:]
    child_pred = j_pred[:, :, 1, :] - j_pred[:, :, 0, :]
    parent_pred = normalize_tensor(parent_pred)
    child_pred = normalize_tensor(child_pred)

    axis_pred = cross_product(child_pred, parent_pred)
    axis_pred = normalize_tensor(axis_pred)
    angle_pred = dot_product(child_pred, parent_pred)
    angle_pred = torch.acos(angle_pred)

    parent_gt = j_gt[:, :, 2, :] - j_gt[:, :, 1, :]
    parent_gt =normalize_tensor(parent_gt)
    child_gt = j_gt[:, :, 1, :] - j_gt[:, :, 0, :]
    child_gt = normalize_tensor(child_gt)
    axis_gt = cross_product(child_gt, parent_gt)
    axis_gt = normalize_tensor(axis_gt)
    angle_gt = dot_product(child_gt, parent_gt)
    angle_gt = torch.acos(angle_gt)

    MSE = torch.nn.MSELoss()

    loss = MSE(angle_pred, angle_gt) + MSE(axis_pred, axis_gt)
    return loss

def angle_loss_2D(j_gt, j_pred):
    angle_list = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 5, 6], [5, 6, 7], [6, 7, 8], [0, 9, 10], [9, 10, 11],
                  [10, 11, 12], [0, 13, 14], [13, 14, 15], [14, 15, 16], [0, 17, 18], [17, 18, 19], [18, 19, 20]]
    j_pred = j_pred[:,angle_list]
    j_gt = j_gt[:,angle_list]

    parent_pred = j_pred[:, :, 2,:] - j_pred[:, :, 1,:]
    child_pred = j_pred[:, :, 1,:] - j_pred[:, :, 0,:]

    parent_pred = normalize_tensor(parent_pred)
    child_pred = normalize_tensor(child_pred)
    angle_pred = dot_product(child_pred, parent_pred)
    angle_pred = torch.acos(angle_pred)

    parent_gt = j_gt[:, :, 2,:] - j_gt[:, :, 1,:]
    child_gt = j_gt[:, :, 1,:] - j_gt[:, :, 0,:]

    parent_gt = normalize_tensor(parent_gt)
    child_gt = normalize_tensor(child_gt)
    angle_gt = dot_product(child_gt, parent_gt)
    angle_gt = torch.acos(angle_gt)

    MSE = torch.nn.MSELoss()
    loss = MSE(angle_gt, angle_pred)
    return loss

def finger_loss_3D(j_gt, j_pred, finger):
    thumb = [0, 13, 14, 15]
    index = [0, 1, 2, 3]
    middle = [0, 4, 5, 6]
    ring = [0, 10, 11, 12]
    little = [0, 7, 8, 9]

    if finger =='thumb' or "thumb":
        gt = j_gt[:,thumb, :]
        pred = j_pred[:, thumb,:]
    elif finger =='index' or "index":
        gt = j_gt[:, index, :]
        pred = j_pred[:, index, :]
    elif finger =='middle' or "middle":
        gt = j_gt[:, middle, :]
        pred = j_pred[:, middle, :]
    elif finger == 'ring' or "ring":
        gt = j_gt[:, ring, :]
        pred = j_pred[:, ring, :]
    elif finger == 'little' or "little":
        gt = j_gt[:, little, :]
        pred = j_pred[:, little, :]
    else:
        print("invalid finger name")
        return

    MSE = torch.nn.MSELoss()
    loss = MSE(gt, pred)
    return loss


def part_joint_loss(j_gt, j_pred):
    # B J 3
    print("")
def reproj_iou_loss(sil_gt, sil_pred):
    #B 128 128
    I = torch.mul(sil_gt, sil_pred)
    I = torch.sum(I, dim= (1,2))
    U = torch.sum(sil_gt, dim=(1,2)) + torch.sum(sil_pred, dim=(1,2)) - I

    reproj_loss = torch.nn.MSELoss()
    loss = reproj_loss(sil_gt, sil_pred) #/U

    return loss

def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src

class Laplacian(torch.autograd.Function):
    def __init__(self, faces):
        # Faces is B x F x 3, cuda torch Variabe.
        # Reuse faces.
        self.F_np = faces.data.cpu().numpy()
        self.F = faces.data
        self.L = None

    def forward(self, V):
        # If forward is explicitly called, V is still a Parameter or Variable
        # But if called through __call__ it's a tensor.
        # This assumes __call__ was used.
        #
        # Input:
        #   V: B x N x 3
        #   F: B x F x 3
        # Outputs: Lx B x N x 3
        #
        # Numpy also doesnt support sparse tensor, so stack along the batch

        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        if self.L is None:
            # Compute
            C = cotangent(V, self.F)
            C_np = C.cpu().numpy()
            batchC = C_np.reshape(-1, 3)
            # Adjust face indices to stack:
            offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
            F_np = self.F_np + offset
            batchF = F_np.reshape(-1, 3)

            rows = batchF[:, [1, 2, 0]].reshape(-1)
            cols = batchF[:, [2, 0, 1]].reshape(-1)
            # Final size is BN x BN
            BN = batchV.shape[0]
            L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN, BN))
            L = L + L.T
            # np.sum on sparse is type 'matrix', so convert to np.array
            M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
            L = L - M
            # remember this
            self.L = L
            # import matplotlib.pylab as plt
            # plt.ion()
            # plt.clf()
            # plt.spy(L)
            # plt.show()
            # import ipdb; ipdb.set_trace()

        Lx = self.L.dot(batchV).reshape(V_np.shape)

        return convert_as(torch.Tensor(Lx), V)

    def backward(self, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = self.L.dot(g_o).reshape(grad_out.shape)

        return convert_as(torch.Tensor(Lg), grad_out)


def cotangent(V, F):
    # Input:
    #   V: B x N x 3
    #   F: B x F  x3
    # Outputs:
    #   C: B x F x 3 list of cotangents corresponding
    #     angles for triangles, columns correspond to edges 23,31,12

    # B x F x 3 x 3
    indices_repeat = torch.stack([F, F, F], dim=2)

    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0])
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1])
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2])

    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area
    A = 2 * torch.sqrt(sp * (sp - l1) * (sp - l2) * (sp - l3))

    cot23 = (l2 ** 2 + l3 ** 2 - l1 ** 2)
    cot31 = (l1 ** 2 + l3 ** 2 - l2 ** 2)
    cot12 = (l1 ** 2 + l2 ** 2 - l3 ** 2)

    # 2 in batch
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    return C

class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces):
        # Input:
        #  faces: B x F x 3
        #import nnutils.laplacian
        # V x V
        self.laplacian = Laplacian(faces)
        self.Lx = None

    def __call__(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = torch.norm(Lx, p=2, dim=1).mean()
        return loss
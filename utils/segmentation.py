import torch
def get_distance_verts2joints(verts, joints):
    batch_size = verts.shape[0]
    dists = torch.zeros([batch_size, 778, 21])
    for vidx in range(778):
        for jidx in range(21):
            dist = verts[:,vidx,:] - joints[:,jidx,:]
            dist = torch.norm(dist, dim = 1)
            dists[:, vidx, jidx] = dist
    return dists

def get_limb_idx(joint1, joint2):
    jidx = torch.where(joint1 > joint2, joint1, joint2)
    jidx = torch.where(((jidx == 1) | (jidx == 5) | (jidx == 9) | (jidx == 13) | (jidx == 17)), torch.ones([1, 778]), jidx)
    return jidx

def check_connection(j1, j2):
    batch_size = j1.shape[0]
    buf_j1 = torch.where(j1 < j2, j2, j1)
    buf_j2 = torch.where(j1 < j2, j1, j2) # batch, n
    j1 = buf_j1
    j2 = buf_j2

    condition1 = (j1-j2) ==1
    condition2 = ((j1 == 5) & (j2 == 1)) | ((j1 == 9) & (j2 == 5)) | ((j1 == 13) & (j2 == 9)) | ((j1 == 17) & (j2 == 13))
    condition3 = ((j1 == 5) & (j2==0)) | ((j1==9) & (j2==0)) | ((j1==13) & (j2==0)) | ((j1==17) & (j2==0))
    condition = condition1 | condition2 | condition3
    mask = torch.where(condition, torch.ones([batch_size, 778]), torch.zeros([batch_size, 778]))
    mask = 2*mask -1 # -1 or 1
    return mask.float()

def get_part_labels(verts, joints):
    batch_size = 1
    dists = get_distance_verts2joints(verts, joints)
    indices = torch.argsort(dists, dim =2, descending=False)
    print(indices.cpu().numpy())
    joint1 = indices[:,:,0].float()
    joint2 = indices[:,:,1].float()
    TF_mask = check_connection(joint1, joint2)
    J_mask = get_limb_idx(joint1, joint2) * (TF_mask)

    J_mask = torch.where(J_mask >=0,  J_mask, torch.zeros([batch_size, 778])) # joint index 1~20, not 0~19, not found == index 0
    return J_mask
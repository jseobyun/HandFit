import matplotlib.pyplot as plt
import os
from config import cfg

def vis_fit_process(img_idx, interval, masks_GT, joints_GT, sil, joints2D_pred, dep):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(masks_GT[0].cpu().numpy(), alpha=0.5)
    ax1.imshow(sil[0].cpu().detach().numpy(), alpha=0.5)
    ax2.imshow(dep[0].cpu().detach().numpy(), cmap='gray', vmin=1, vmax=3)

    joints2D_buf = joints2D_pred.detach().cpu()
    jointsGT_buf = joints_GT.detach().cpu()

    joints = 224 * joints2D_buf[0, :, :].numpy()
    joint_GT = 224 * jointsGT_buf[0, :, :].numpy()

    ax1.scatter(joint_GT[:, 0], joint_GT[:, 1], color='k', marker='.')
    ax1.scatter(joints[:, 0], joints[:, 1], color='r', marker='.')

    ax1.plot(joint_GT[0:2, 0], joint_GT[0:2, 1], color='r')  # 01
    ax1.plot(joint_GT[1:3, 0], joint_GT[1:3, 1], color='r')  # 12
    ax1.plot(joint_GT[2:4, 0], joint_GT[2:4, 1], color='r')  # 23
    ax1.plot(joint_GT[3:5, 0], joint_GT[3:5, 1], color='r')  # 34

    ax1.plot([joint_GT[5, 0], joint_GT[0, 0]], [joint_GT[5, 1], joint_GT[0, 1]], color='g')
    ax1.plot(joint_GT[5:7, 0], joint_GT[5:7, 1], color='g')  # 56
    ax1.plot(joint_GT[6:8, 0], joint_GT[6:8, 1], color='g')  # 67
    ax1.plot(joint_GT[7:9, 0], joint_GT[7:9, 1], color='g')  # 78

    ax1.plot([joint_GT[9, 0], joint_GT[0, 0]], [joint_GT[9, 1], joint_GT[0, 1]], color='b')
    ax1.plot(joint_GT[9:11, 0], joint_GT[9:11, 1], color='b')  # 9 10
    ax1.plot(joint_GT[10:12, 0], joint_GT[10:12, 1], color='b')  # 10 11print(theta[:,0:4])
    ax1.plot(joint_GT[11:13, 0], joint_GT[11:13, 1], color='b')  # 11 12

    ax1.plot([joint_GT[13, 0], joint_GT[0, 0]], [joint_GT[13, 1], joint_GT[0, 1]], color='c')
    ax1.plot(joint_GT[13:15, 0], joint_GT[13:15, 1], color='c')  # 13 14
    ax1.plot(joint_GT[14:16, 0], joint_GT[14:16, 1], color='c')  # 14 15
    ax1.plot(joint_GT[15:17, 0], joint_GT[15:17, 1], color='c')  # 15 16

    ax1.plot([joint_GT[17, 0], joint_GT[0, 0]], [joint_GT[17, 1], joint_GT[0, 1]], color='m')
    ax1.plot(joint_GT[17:19, 0], joint_GT[17:19, 1], color='m')  # 13 14
    ax1.plot(joint_GT[18:20, 0], joint_GT[18:20, 1], color='m')  # 14 15
    ax1.plot(joint_GT[19:21, 0], joint_GT[19:21, 1], color='m')  # 15 16

    plt.savefig(os.path.join(cfg.vis_dir, "img{}_iter{}.png".format(img_idx, interval)))
    plt.close()
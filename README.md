Hand fitting (MANO)
======
place holder

## Installation

MANO : https://github.com/hassony2/manopth

neural-renderer-pytorch : https://github.com/daniilidis-group/neural_renderer

pyquaternion : https://github.com/KieranWynn/pyquaternion

chumpy : https://github.com/mattloper/chumpy

opencv

## Note

Put these function to manolayer.py

    def pose_pca2quat(self, th_pose_coeffs):
        batch_size = th_pose_coeffs.shape[0]
        th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                         self.ncomps].cuda()
        # print("th_hand_pose_coeeffs", np.shape(th_hand_pose_coeffs))
        # print("th_selected_comps", np.shape(self.th_selected_comps))
        th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps.cuda())
        inverse_mat = torch.inverse(self.th_selected_comps)
        a = th_full_hand_pose.mm(inverse_mat.cuda())

        th_full_pose = torch.cat([th_pose_coeffs[:, :self.rot].cuda(), self.th_hands_mean.cuda() + th_full_hand_pose], 1)
        th_full_pose = th_full_pose.view(batch_size, -1, 3)

        Qs = rodrigues_layer.batch_quaternion(th_full_pose.reshape(-1,3)).reshape(-1, 16, 4)
        return Qs

    def pose_quat2pca(self,th_pose_quat_coeffs):
        batch_size = th_pose_quat_coeffs.shape[0]
        quats = th_pose_quat_coeffs.reshape([batch_size, 16, 4])

        arccos = torch.acos(quats[:,:,0:1]).repeat(1,1,3)
        # arccos = torch.cat([arccos,arccos, arccos], dim=2)
        sin = torch.sin(arccos)
        PCA = quats[:,:,1:4]/sin

        PCA = (PCA*(2*arccos)).reshape([-1,48])
        PCA[:,3:] = PCA[:,3]- self.th_hands_mean.cuda()

        # PCA = PCA.reshape([batch_size, 45])
        inverse_mat = torch.inverse(self.th_selected_comps)
        PCA[:,3:] = PCA[:,3:].mm(inverse_mat.cuda())

        return PCA

    def pose_pca2Rotmat(self, th_pose_coeffs):
        batch_size = th_pose_coeffs.shape[0]
        th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                         self.ncomps].cuda()
        th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps.cuda())
        th_full_pose = torch.cat([th_pose_coeffs[:, :self.rot].cuda(), self.th_hands_mean.cuda() + th_full_hand_pose], 1)
        th_full_pose = th_full_pose.view(batch_size, -1, 3)

        Rs = rodrigues_layer.batch_rodrigues(th_full_pose.reshape(-1,3)).reshape(-1, 16, 3,3)
        return Rs

Comment these lines in manolayer.py

    # Scale to milimeters
    # th_verts = th_verts * 1000
    # th_jtr = th_jtr * 1000

## Usage

    python main.py
Visualization of fitting process is saved in output/vis 


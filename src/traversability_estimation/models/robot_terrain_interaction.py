import torch
from torch import nn
import numpy as np
from torchdiffeq import odeint, odeint_adjoint
import matplotlib.pyplot as plt


class RigidBodySoftTerrain(nn.Module):

    def __init__(self, height=np.zeros([8, 3]),
                 terrain_predictor=None,
                 damping=5.0, elasticity=50.0,
                 mass=10.0, gravity=9.8,
                 init_pos_x=[0.5, 0.5, 1.0],
                 init_pos_rpy=[0., 0., 0.],
                 init_vel_x=[2., 0., 0.],
                 init_vel_omega=[0., 0., 0.],
                 adjoint=False,
                 device=torch.device('cpu')):
        super().__init__()
        self.gravity = nn.Parameter(torch.as_tensor([gravity]))
        self.grid_res = 0.1
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.height0 = torch.tensor(height, device=device)
        self.height = torch.tensor(height, device=device)
        # Height map is a learnable parameter if there is no terrain prediction model.
        # Otherwise, if it is provided, its network parameters are being learned.
        if terrain_predictor is None:
            self.height = nn.Parameter(self.height)
        self.damping = nn.Parameter(torch.ones_like(self.height) * damping)
        self.elasticity = nn.Parameter(torch.ones_like(self.height) * elasticity)

        self.terrain_predictor = terrain_predictor

        self.init_pos_x = nn.Parameter(torch.tensor(init_pos_x).view((3, 1)))
        self.init_pos_R = nn.Parameter(self.rpy2rot(*init_pos_rpy))

        self.init_vel_x = nn.Parameter(torch.tensor(init_vel_x).view((3, 1)))  # linear velocity of cog
        self.init_vel_omega = nn.Parameter(torch.tensor(init_vel_omega).view((3, 1)))  # angular (axis-angle) velocity of cog

        # dx = (torch.tensor([0.0]))  # location of points wrt cog
        # dy = (torch.tensor([0.0]))  # location of points wrt cog
        # dz = (torch.tensor([0.0]))  # location of points wrt cog
        dx = (torch.tensor([-0.5, +0.0, +0.5, +0.5, -0.5, +0.0]))  # location of points wrt cog
        dy = (torch.tensor([-0.3, -0.3, -0.3, +0.3, +0.3, +0.3]))  # location of points wrt cog
        dz = (torch.tensor([+0.0, +0.0, +0.0, +0.0, +0.0, +0.0]))  # location of points wrt cog
        self.d = nn.Parameter(torch.stack((dx, dy, dz)))
        self.f = nn.Parameter(torch.zeros_like(self.d))

        self.mass = nn.Parameter(torch.tensor([mass]))
        self.inertia_inv = 1 * torch.tensor([[0.1, 0, 0], [0, 0.04, 0], [0, 0, 0.04]], device=device)
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state):
        pos_x, pos_R, vel_x, vel_omega, f_old = state

        dpos_x = vel_x
        vel_omega_skew = torch.tensor([[0., -vel_omega[2], vel_omega[1]],
                                       [vel_omega[2], 0., -vel_omega[0]],
                                       [-vel_omega[1], vel_omega[0], 0.]]).to(vel_omega.device)
        dpos_R = vel_omega_skew @ pos_R
        points = pos_R @ self.d + pos_x
        dpoints = vel_omega_skew @ (points - pos_x) + vel_x

        if self.terrain_predictor is not None:
            self.height = self.terrain_predictor(self.height0[None][None]).squeeze()

        # interpolate
        h = self.interp(self.height, points[0:2, :])
        e = self.interp(self.elasticity, points[0:2, :])
        d = self.interp(self.damping, points[0:2, :])

        # contacts
        contact = (points[2, :] <= h)

        # Compute terrain + gravity forces
        z = torch.tile(torch.tensor([[0], [0], [1]]), (1, points.shape[1])).to(points.device)
        f = (z * (e * (h - points[2, :]) - d * dpoints[2, :])) * contact  # (nh * dpoints).sum(dim=0)
        fg = self.mass * self.gravity  # * (points[2, :] >= h)
        f[2, :] = f[2, :] - fg  # * (1 - contact.float())

        # Accelerations: linear and angular accelerations computed from forces
        dvel_x = torch.stack(((f[0, :].sum() / self.mass).squeeze(),
                              (f[1, :].sum() / self.mass).squeeze(),
                              (f[2, :].sum() / self.mass).squeeze()))
        dvel_omega = self.inertia_inv @ torch.cross(self.d, f).sum(dim=1)

        return dpos_x, dpos_R, dvel_x, dvel_omega, f  # _track #torch.zeros_like(self.f)

    def get_initial_state(self):
        state = (self.init_pos_x, self.init_pos_R, self.init_vel_x, self.init_vel_omega, self.f)
        return self.t0, state

    def visu(self, x, col='b'):
        plt.clf()
        pos_x, pos_R, vel_x, vel_omega, f = x
        plt.plot(pos_x[:, 0].detach().numpy(), pos_x[:, 2].detach().numpy(), 'o--', color='k', linewidth=2.0)
        points = pos_R @ self.d + pos_x
        # for p in range(4):
        #     plt.plot(points[:, 0, p].detach().numpy(), points[:, 2, p].detach().numpy(), 'o--', linewidth=2.0)
        h = self.height[:, 0].detach().numpy().squeeze()
        # plt.bar(np.linspace(0, h.shape[0]-1, h.shape[0]), h)
        plt.plot(np.linspace(0, h.shape[0] - 1, h.shape[0]), h, color='b', linewidth=5)
        plt.axis('equal')
        plt.grid()
        cols = ['c', 'r', 'c', 'y']
        for t in range(points.shape[0]):
            plt.clf()
            # plt.bar(np.linspace(0, h.shape[0] - 1, h.shape[0]), h)
            plt.plot(np.linspace(0, h.shape[0] - 1, h.shape[0]), h, color='b', linewidth=5)
            plt.axis('equal')
            plt.grid()
            for p in range(2):
                plt.plot([points[t, 0, p].detach().numpy(), points[t, 0, (p+1)%4].detach().numpy()], [points[t, 2, p].detach().numpy(), points[t, 2, (p+1)%4].detach().numpy()], '.-', color=cols[p], linewidth=5.0)
            plt.pause(0.01)

        # for t in range(tt.shape[0]):
        #     plt.plot([x1[t].detach().numpy(), x2[t].detach().numpy()], [y1[t].detach().numpy(), y2[t].detach().numpy()], '.-', color=col, linewidth=2.0)
        #     #plt.pause(0.01)

    def rpy2rot(self, roll, pitch, yaw):
        roll = torch.as_tensor(roll)
        pitch = torch.as_tensor(pitch)
        yaw = torch.as_tensor(yaw)
        RX = torch.tensor([[1, 0, 0],
                           [0, torch.cos(roll), -torch.sin(roll)],
                           [0, torch.sin(roll), torch.cos(roll)]])

        RY = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                           [0, 1, 0],
                           [-torch.sin(pitch), 0, torch.cos(pitch)]])

        RZ = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                           [torch.sin(yaw), torch.cos(yaw), 0],
                           [0, 0, 1]])
        return RZ @ RY @ RX

    def interp(self, f, pt, mode='bilinear'):
        # example:
        # im = torch.rand((4,8)).view(1,1,4,8)
        # pt = torch.tensor([[2, 2.25, 2.5, 2.75, 3,4],[1.5,1.5,1.5,1.5,1.5,1.5]], dtype=torch.double)
        H = f.shape[0]
        W = f.shape[1]
        WW = (W - 1) / 2
        HH = (H - 1) / 2
        pt_r = pt.clone()
        pt_r[1, ] = (pt[0, :] - HH) / HH
        pt_r[0, ] = (pt[1, :] - WW) / WW
        return torch.nn.functional.grid_sample(f.view(1, 1, H, W), pt_r.permute(1, 0).view(1, 1, pt_r.shape[1], pt_r.shape[0]), mode=mode, align_corners=True).squeeze()


def main():
    VISU = 2
    OUTPUT_PATH = '/home/ruslan/workspaces/traversability_ws/src/traversability_estimation/data/output'
    CREATE_MOVIE = 1
    torch.set_default_dtype(torch.float64)

    if VISU == 2:
        from mayavi import mlab

    height = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
                       [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
                       [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5],
                       [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
                       [0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5],
                       [0.5, 0.5, 0.0, 0.5, 0.7, 0.5, 0.5, 0.0, 0.5, 0.7],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    system_true = RigidBodySoftTerrain(height=height, damping=300.0, elasticity=5000.0, mass=20.0)

    total_time = 4.0
    number_of_samples = 100
    t0, state = system_true.get_initial_state()
    tt = torch.linspace(float(t0), total_time, number_of_samples)
    x_traj = odeint(system_true.forward, state, tt, atol=1e-3, rtol=1e-3)

    if VISU == 1:
        system_true.visu(x_traj)
    elif VISU == 2:
        pos_x, pos_R, vel_x, vel_omega, aux = x_traj
        h = system_true.height.detach().numpy()
        x_grid, y_grid = np.mgrid[0:system_true.height.shape[0], 0:system_true.height.shape[1]]
        points = system_true.d.detach().numpy()
        for t in range(pos_x.shape[0]):
            _, _, _, _, f = system_true.forward(tt[t], (pos_x[t], pos_R[t], vel_x[t], vel_omega[t], aux[t]))

            mlab.clf()
            mlab.plot3d(pos_x[0:(t + 1), 0].detach().numpy(), pos_x[0:(t + 1), 1].detach().numpy(),
                        pos_x[0:(t + 1), 2].detach().numpy(), color=(1, 1, 1), line_width=2.0)
            mlab.surf(x_grid, y_grid, h, opacity=0.5, representation='wireframe', line_width=5.0)
            mlab.surf(x_grid, y_grid, h, color=(0.5, 0.5, 1.0), opacity=0.5, representation='surface')
            cog = pos_x[t].detach().numpy()
            rot = pos_R[t].detach().numpy()
            d = rot @ points
            mlab.points3d(cog[0] + d[0, :], cog[1] + d[1, :], cog[2] + d[2, :], scale_factor=0.25)
            mlab.quiver3d(cog[0] + d[0, :], cog[1] + d[1, :], cog[2] + d[2, :], f[0].detach().numpy(),
                          f[1].detach().numpy(), f[2].detach().numpy(), scale_factor=0.005)
            # robot.plot_robot_Rt([], pos_R[t].detach().numpy(), pos_x[t].detach().numpy())
            mlab.view(azimuth=150 - t, elevation=80, distance=12.0)
            if CREATE_MOVIE:
                mlab.savefig(filename=OUTPUT_PATH + '{:04d}_frame'.format(t) + '.png', magnification=1.0)
            # else:
            #    mlab.show()


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt

from common.utils import generate_eom


class Plant(object):

    def __init__(self, model=None):
        super().__init__()
        self.model = model
    
    def get_T(self):
        T = 1/2*(self.qdot.T)@ self.M @ self.qdot
        return T.squeeze()

    def get_V(self):
        return 3/2*np.cos(self.q[0]) + 1/2*np.cos(self.q[0]+self.q[1])*9.8
    
    def get_E(self):
        return self.get_T() + self.get_V()
    
    def update_state(self, q, qdot):

        self.q = q
        self.qdot = qdot

        self.M, self.c, self.g = generate_eom(q, qdot)

        self.T = self.get_T()
        self.V = self.get_V()
        self.E = self.get_E()
    
    def calc_next_state(self, q, qdot, qddot, dt=0.01):
        q = q + qdot*dt
        qdot = qdot + qddot*dt
        return q, qdot
    
    def calc_ik(self, q, qdot, qddot):
        tau = self.M@qddot.reshape(-1,1) + self.c + self.g
        return tau
    
    def calc_fk(self, q, qdot, u):
        qddot = np.linalg.inv(self.M)@(u - self.c - self.g)
        return qddot
    
    def transit(self, u):
        qddot = self.calc_fk(self.q, self.qdot, u)
        q_next, qdot_next = self.calc_next_state(self.q, self.qdot, qddot)
        self.update_state(q_next, qdot_next)
    
    def init_state(self, q, qdot):
        self.update_state(q, qdot)


class Visualizer(object):

    def __init__(self):
        super().__init__()
        self.u = {'pred':[], 'gt':[]}
        self.M = {'pred':[], 'gt':[]}
        self.c = {'pred':[], 'gt':[]}
        self.g = {'pred':[], 'gt':[]}
        self.q = []
        self.qdot = []
        self.qddot = []
        self.stacked = False
    
    def add_data(self, q, qdot, qddot, u, M, c, g):
        self.u['pred'].append(u[0]); self.u['gt'].append(u[1])
        self.M['pred'].append(M[0]); self.M['gt'].append(M[1])
        self.c['pred'].append(c[0]); self.c['gt'].append(c[1])
        self.g['pred'].append(g[0]); self.g['gt'].append(g[1])
        self.q.append(q)
        self.qdot.append(qdot)
        self.qddot.append(qddot)
    
    def stack(self):
        self.u['pred'], self.u['gt'] = np.stack(self.u['pred']).squeeze(), np.stack(self.u['gt']).squeeze()
        self.M['pred'], self.M['gt'] = np.stack(self.M['pred']).squeeze(), np.stack(self.M['gt']).squeeze()
        self.c['pred'], self.c['gt'] = np.stack(self.c['pred']).squeeze(), np.stack(self.c['gt']).squeeze()
        self.g['pred'], self.g['gt'] = np.stack(self.g['pred']).squeeze(), np.stack(self.g['gt']).squeeze()
        self.q = np.stack(self.q).squeeze()
        self.qdot = np.stack(self.qdot).squeeze()
        self.qddot = np.stack(self.qddot).squeeze()
        self.stacked = True

    def save_plot(self, fmt='pdf'):
        assert fmt in ['pdf', 'png']
        file_name = 'rst.' + fmt

        # stack datasets
        if not self.stacked:
            self.stack()

        # plot
        plt.rcParams["legend.frameon"] = False
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.size"] = 10   

        fig, axes = plt.subplots(2, 4, sharex=True, figsize=(24.0, 6.0))

        for i in range(2):
            axes[i,0].plot(self.u['gt'][:,i], label='Ground Truth',c='k')
            axes[i,0].plot(self.u['pred'][:,i], label='Predicted',c='r', alpha=0.8)
            axes[i,1].plot(self.M['gt'][:,i], label='Ground Truth',c='k')
            axes[i,1].plot(self.M['pred'][:,i], label='Predicted',c='r', alpha=0.8)
            axes[i,2].plot(self.c['gt'][:,i], label='Ground Truth',c='k')
            axes[i,2].plot(self.c['pred'][:,i], label='Predicted',c='r', alpha=0.8)
            axes[i,3].plot(self.g['gt'][:,i], label='Ground Truth',c='k')
            axes[i,3].plot(self.g['pred'][:,i], label='Predicted',c='r', alpha=0.8)

        # set title
        axes[0,0].set_title(r'$\mathbf{\tau}$', fontsize=16, y=1.05)
        axes[0,1].set_title(r'$\mathbf{H}(\mathbf{q}) \ddot{\mathbf{q}}$', fontsize=16, y=1.05)
        axes[0,2].set_title(r'$\mathbf{c}(\mathbf{q}, \mathbf{\dot{q}})$', fontsize=16, y=1.05)
        axes[0,3].set_title(r'$\mathbf{g}(\mathbf{q})$', fontsize=16, y=1.05)

        axes[0,0].set_ylabel('Joint 0\nTorque', fontsize=12)
        axes[1,0].set_ylabel('Joint 1\nTorque', fontsize=12)

        handles, labels = [], []
        for i in range(2):
            for j in range(4):
                ax = axes[i,j]
                ax.set_ylim(axes[i,0].get_ylim())

                h, l = ax.get_legend_handles_labels()
                handles.append(h)
                labels.append(l)

        plt.legend(['Ground Truth', 'Predicted'], loc='upper right', bbox_to_anchor=(1.0, -.15), ncol=2, fontsize=12)
        fig.subplots_adjust(top=0.8, hspace=0.8)

        fig.align_ylabels()

        fig.tight_layout()
        plt.savefig(file_name)
        print('saved: {}'.format(file_name))
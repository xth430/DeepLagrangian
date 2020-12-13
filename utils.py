import numpy as np


def generate_eom(q, qdot):
    """E.O.M of double pendulum
        u = M(q)*qddot + c(q,qdot)*qdot + g(q)  

    Arguments:
        state {[ndarray]} -- [l*2]
        state_d {[ndarray]} -- [l*2]

    return:
        M, c, g
    """ 

    M = np.array([
        [5/3 + np.cos(q[1]), 1/3 + 1/2*np.cos(q[1])],
        [1/3 + 1/2*np.cos(q[1]), 1/3                 ]
    ])

    c = np.array([
        [-1/2*(2*qdot[0]*qdot[1] + qdot[1]**2)*np.sin(q[1])],
        [1/2*(qdot[0]**2)*np.sin(q[1])] 
    ])

    g = np.array([
        [-3/2 * np.sin(q[0]) - 1/2*np.sin(q[0]+q[1])],
        [-1/2 * np.sin(q[0] + q[1])]
    ]) * 9.8 * 50

    return M, c, g


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def split_states(states):
#     if states.ndim == 2:
#         s, s_d, s_dd = states[:,0:1], states[:,1:2], states[:,2:3]
#     if states.ndim == 3:
#         s, s_d, s_dd = states[:,:,0:1], states[:,:,1:2], states[:,:,2:3]
#     return s, s_d, s_dd

def split_states(states):
    return states[...,0], states[...,1], states[...,2]
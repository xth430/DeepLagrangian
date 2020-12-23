import numpy as np
import torch

def generate_eom(q, qdot):

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
    ]) * 9.8

    return M, c, g


def split_states(states):
    return states[...,0], states[...,1], states[...,2]


class Datasets(torch.utils.data.Dataset):
    def __init__(self, states, targets):
        assert len(states) == len(targets)
        self.data_num = len(targets)
        self.states = states
        self.targets = targets

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]


def gen_loader(states, targets, batch_size=1, shuffle=False):
    states, targets = torch.tensor(states), torch.tensor(targets)
    datasets = Datasets(states, targets)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)

    return train_loader


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
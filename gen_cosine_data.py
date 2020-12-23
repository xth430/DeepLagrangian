import os

import numpy as np

from common.utils import generate_eom, split_states
from common.obj import Plant


# config
T = 1.0
dt = 0.01


train_args = [
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[2, 5], 'b': 1.0},
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[2, 7], 'b': 1.0},
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[3, 5], 'b': 1.0},
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[3, 7], 'b': 1.0},
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[3, 11], 'b': 1.0},
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[5, 11], 'b': 1.0},
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[5, 13], 'b': 1.0}
]
test_args = [
    {'A':[0.3*np.pi, 0.5*np.pi], 'w':[5, 3], 'b': 1.0}
]


def _cosine_func(arg):
    A, w, beta = np.array(arg['A']), np.array(arg['w']), np.array(arg['b'])
    return lambda t: A*np.cos(2*np.pi*w*beta*t + np.sin(0.5*np.pi*t))


def _gen_cosine_sequence(arg, T=T, dt=dt):
    l = int(T/dt)
    sequence = []
    func = _cosine_func(arg)
    for i in range(l):
        sequence.append(func(i*dt))
    sequence = np.stack(sequence, axis=0)

    return sequence


def diff(x, dt=dt):
    return (x[2:] - x[:-2]) / 2 / dt


def gen_cosine_states(args):
    states = []
    for arg in args:
        q = _gen_cosine_sequence(arg)
        q_t = diff(q)
        q_tt = diff(q_t)
        states.append(np.stack([q[2:-2], q_t[1:-1], q_tt], axis=2))
    states = np.concatenate(states, axis=0)
    return states


def generate_cosine_datasets(phase='train'):

    if not os.path.exists('data'):
        os.mkdir('data')

    assert phase == 'train' or phase == 'test'

    states = gen_cosine_states(train_args if phase=='train' else test_args)
    targets = []

    p = Plant()
    for l in range(states.shape[0]):
        q, qdot, qddot = split_states(states[l])
        p.update_state(q, qdot)
        u = p.calc_ik(q, qdot, qddot)
        targets.append(u.squeeze())
    targets = np.stack(targets)

    np.save('data/{}_states_cosine.npy'.format(phase), states)
    np.save('data/{}_torque_cosine.npy'.format(phase), targets)
    
    print('>>> {} data done!'.format(phase))


if __name__ == "__main__":
    generate_cosine_datasets('train')
    generate_cosine_datasets('test')
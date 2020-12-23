import os 

import numpy as np 
import matplotlib.pyplot as plt  

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.utils import AverageMeter, split_states, generate_eom, gen_loader
from common.model import DeLaN, init_weights
from common.obj import Visualizer

import argparse 

parser = argparse.ArgumentParser()
# model argument
parser.add_argument('--inp_dim', type=int, default=2) 
parser.add_argument('--hid_dim', type=int, default=128) 
parser.add_argument('--num_layers', type=int, default=4) 
parser.add_argument('--bias', type=float, default=1e-4) 
# training argument
parser.add_argument('-e','--epochs', type=int, default=200) 
parser.add_argument('--lr', type=float, default=0.005) 
parser.add_argument('--T', type=float, default=1.0) 
parser.add_argument('--dt', type=float, default=0.01) 

args = parser.parse_args()


# read datasets
datasets = 'cosine'
train_states = np.load(os.path.join('data', 'train_states_{}.npy'.format(datasets)))
train_torque = np.load(os.path.join('data', 'train_torque_{}.npy'.format(datasets)))
test_states = np.load(os.path.join('data', 'test_states_{}.npy'.format(datasets)))
test_torque = np.load(os.path.join('data', 'test_torque_{}.npy'.format(datasets)))

train_loader = gen_loader(train_states, train_torque, batch_size=64, shuffle=True)
test_loader = gen_loader(test_states, test_torque, batch_size=1, shuffle=False)


def main(args):
    print('>>> Loading model...')
    model = DeLaN(args)
    model.apply(init_weights)
    print('>>> total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.92)

    print('\n>>> start train')
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, scheduler, epoch)
    
    print('\n>>> start evaluate')
    evaluate(test_loader, model, criterion)


def train(data_loader, model, criterion, optimizer, scheduler, epoch):
    running_loss = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model.train()

    for i, (state, target) in enumerate(data_loader):

        state = state.float(); target = target.float()
        num_states = state.shape[0]

        pred, M, c, g = model(state) # pred.shape: (batch_size, 2)

        loss = criterion(pred, target) / 1000000

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item(), num_states)
    
    print('[{}] loss: {:.3f}, lr: {:.5f}'.format(epoch+1, running_loss.avg, scheduler.get_last_lr()[0]))

    scheduler.step()


def evaluate(data_loader, model, criterion):
    running_loss = AverageMeter()

    # Switch to eval mode
    model.eval()

    # visualizer
    viz = Visualizer()

    for i, (state, target) in enumerate(data_loader):

        state = state.float(); target = target.float() # state.shape: (batch_size, 2, 3)
        num_states = state.shape[0]

        pred, M, c, g = model(state) # pred.shape: (batch_size, 2)

        loss = criterion(pred, target) / 1000000

        running_loss.update(loss.item(), num_states)

        # test
        q, qdot, qddot = split_states(state.numpy().squeeze())
        M_gt, c_gt, g_gt = generate_eom(q, qdot)

        M_pred = M.detach().numpy()
        c_pred = c.detach().numpy()
        g_pred = g.detach().numpy()

        u_pred, u_gt = pred.detach().numpy(), target.detach().numpy()

        viz.add_data(q, qdot, qddot, 
            (u_pred, u_gt),
            (M_pred @ qddot.reshape(2,), M_gt @ qddot.reshape(2,)),
            (c_pred, c_gt),
            (g_pred, g_gt)
        )

    print('evaluate loss: {:.3f}'.format(running_loss.avg))
    
    viz.save_plot()


if __name__ == "__main__":
    main(args)
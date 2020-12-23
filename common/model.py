import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.gradcheck import zero_gradients

from common.utils import split_states


class FCNet(nn.Module):

    def __init__(self, inp_dim, hid_dim, num_layers, bias):
        super(FCNet,self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.low_dim = ((self.inp_dim-1)*self.inp_dim)//2  # dim of lower triangular matrix
        self.num_layers = num_layers
        self.bias = bias

        self.pre = nn.Linear(self.inp_dim, self.hid_dim)
        self.fc_list = nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.num_layers)])
        self.fc_lo = nn.Linear(self.hid_dim, self.low_dim)
        self.fc_ld = nn.Linear(self.hid_dim, self.inp_dim)
        self.fc_g = nn.Linear(self.hid_dim, self.inp_dim)
        self.nonlinearity = nn.LeakyReLU()

    def forward(self, q):
        # forward propagation
        x = self.nonlinearity(self.pre(q))
        for i in range(self.num_layers):
            x = self.fc_list[i](x)
            x = self.nonlinearity(x)
        
        lo = self.fc_lo(x)
        ld = nn.Softplus()(self.fc_ld(x)) + self.bias
        g = self.fc_g(x)

        return lo, ld, g


class DeLaN(nn.Module):

    def __init__(self, args):
        super(DeLaN, self).__init__()
        self.inp_dim = args.inp_dim
        self.lo_indices = torch.tril_indices(row=self.inp_dim, col=self.inp_dim, offset=-1)
        self.ld_indices = [range(self.inp_dim), range(self.inp_dim)]

        self.fcnet = FCNet(args.inp_dim, args.hid_dim, args.num_layers, args.bias)

    def forward(self, state):
        # generate input parameters
        q, qdot, qddot = split_states(state)
        q.requires_grad=True
        q.retain_grad()
        batch_size = q.shape[0]

        # forward prop
        lo, ld, g = self.fcnet(q)

        # generate matrix H
        L = torch.zeros(batch_size, self.inp_dim, self.inp_dim)
        L[:, self.lo_indices[0], self.lo_indices[1]] = lo
        L[:, self.ld_indices[0], self.ld_indices[1]] = ld
        H = L @ L.transpose(-2,-1)

        # calc partial deriv
        lo_q = compute_jacobian(q, lo)
        ld_q = compute_jacobian(q, ld)

        L_q = torch.zeros(batch_size, self.inp_dim, self.inp_dim, self.inp_dim)
        L_q[:,self.lo_indices[0], self.lo_indices[1]] = lo_q
        L_q[:,self.ld_indices[0], self.ld_indices[1]] = ld_q

        # calculate Hdot
        Ldot = torch.einsum('bijk,bk->bij',L_q, qdot)
        Hdot = L @ Ldot.transpose(-2,-1) + Ldot @ L.transpose(-2,-1)

        # calculate dH/dt * qdot
        c1 = (Hdot @ qdot.unsqueeze(-1)).squeeze()

        mid = torch.einsum('bijk,blj->bilk', L_q, L) + torch.einsum('bkjl,bij->bikl', L_q, L)
        c2 = torch.einsum('bi,bijk,bj->bk', qdot, mid, qdot)

        # c(q,qdot)
        c = c1 - 1/2*c2

        u = (H @ qddot.unsqueeze(-1)).squeeze() + c + g

        return u, H, c, g


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X out_dims
    :return: jacobian: Batch X out_dims X Size
    """
    assert inputs.requires_grad

    out_dims = output.size()[1]

    jacobian = torch.zeros(out_dims, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(out_dims):
        zero_gradients(inputs)
        grad_output = torch.zeros(*output.size())
        grad_output[:, i] = 1
        grads = torch.autograd.grad(outputs=output, inputs=inputs, grad_outputs=grad_output, create_graph=True, retain_graph=True)
        jacobian[i] = grads[0]

    return torch.transpose(jacobian, dim0=0, dim1=1)
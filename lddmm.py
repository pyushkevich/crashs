# Kernel definitions (from KeOps)
import numpy as np
import torch
from pykeops.torch import Vi, Vj
from torch.autograd import grad

# For LDDMM
def GaussKernel(sigma):
    x, y, b = Vi(0, 3), Vj(1, 3), Vj(2, 3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp()
    return (K * b).sum_reduction(axis=1)

# For Varifold
def GaussLinKernel(sigma):
    x, y, u, v, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vj(4, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum() ** 2
    return (K * b).sum_reduction(axis=1)

# For Varifold
def GaussLinKernelWithLabels(sigma, nlabels):
    x, y, u, v, lx, ly, b = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vi(4, nlabels), Vj(5, nlabels), Vj(6, 1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum() ** 2 * (lx * ly).sum()
    return (K * b).sum_reduction(axis=1)

# For Currents
def GaussLinCurrentsKernel(sigma):
    x, y, u, v = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum()
    return K.sum_reduction(axis=1)

# For Currents
def GaussLinCurrentsKernelWithLabels(sigma, nlabels):
    x, y, u, v, lx, ly = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3), Vi(4, 5), Vj(5, 5)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum() * (lx * ly).sum()
    return K.sum_reduction(axis=1)

# For Currents to match C++ code
def GaussLinCurrentsKernelC(sigma):
    x, y, u, v = Vi(0, 3), Vj(1, 3), Vi(2, 3), Vj(3, 3)
    gamma = 1 / (2 * sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2 * gamma).exp() * (u * v).sum() * 0.5
    return K.sum_reduction(axis=1)

# Forward integration
def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                map(
                    lambda x, xdot, xdoti: x + (0.25 * dt) * (xdot + 3 * xdoti),
                    x,
                    xdot,
                    xdoti,
                )
            )
            l.append(x)
        return l

    return f

# LDDMM definitions

def Hamiltonian(K):
    def H(p, q):
        return 0.5 * (p * K(q, q, p)).sum()

    return H


def HamiltonianSystem(K):
    H = Hamiltonian(K)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS

def Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt)


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)

    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)

    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K, dataloss, nt=10, gamma=0):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K, nt)[-1]
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)

    return loss


# Basic Varifold loss
# VT: vertices coordinates of target surface,
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossVarifoldSurf(FS, VT, FT, K):
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return (
            cst
            + (LS * K(CS, CS, NSn, NSn, LS)).sum()
            - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()
        )

    return loss


# Basic Varifold loss with labels
# VT: vertices coordinates of target surface,
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossVarifoldSurfWithLabels(FS, VT, FT, lab_S, lab_T, K):
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K(CT, CT, NTn, NTn, lab_T, lab_T, LT)).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return (
            cst
            + (LS * K(CS, CS, NSn, NSn, lab_S, lab_S, LS)).sum()
            - 2 * (LS * K(CS, CT, NSn, NTn, lab_S, lab_T, LT)).sum()
        )

    return loss


# Also implement a basic currents loss
def lossCurrentsSurf(FS, VT, FT, K):
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        return centers, normals

    CT, NT = get_center_length_normal(FT, VT)
    cst = K(CT, CT, NT, NT).sum()
    
    def loss(VS):
        CS, NS = get_center_length_normal(FS, VS)
        tt,ss,st = cst, K(CS, CS, NS, NS).sum(), 2 * K(CS, CT, NS, NT).sum()
        print(f'tt: {tt}, ss: {ss}, st: {st}')
        return (
            cst
            + K(CS, CS, NS, NS).sum()
            - 2 * K(CS, CT, NS, NT).sum()
        )

    return loss


# Basic Varifold loss with labels
# VT: vertices coordinates of target surface,
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossCurrentsSurfWithLabels(FS, VT, FT, lab_S, lab_T, K):
    def get_center_length_normal(F, V):
        V0, V1, V2 = (
            V.index_select(0, F[:, 0]),
            V.index_select(0, F[:, 1]),
            V.index_select(0, F[:, 2]),
        )
        centers, normals = (V0 + V1 + V2) / 3, 0.5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals**2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals / length

    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = K(CT, CT, NTn, NTn, lab_T, lab_T).sum()

    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return (
            cst
            + K(CS, CS, NSn, NSn, lab_S, lab_S).sum()
            - 2 * K(CS, CT, NSn, NTn, lab_S, lab_T).sum()
        )

    return loss



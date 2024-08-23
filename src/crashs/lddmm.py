# Kernel definitions (from KeOps)
import numpy as np
import torch
import time
import geomloss
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

def rotation_from_vector(x):
    """
    Generate a 3D rotation vector from three parameters.

    Args:
        x: 
            A torch tensor of shape (3). It contains the parameters of the rotation.
            [Write more detail about what the parameters mean geometrically]
    Output:
        A shape (3,3) tensor holding a rotation matrix corresponding to x
    """
    # I will use the the axis/angle representation. The norm of the vector x gives the
    # angle in radians, and the normalized vector is the axis around which the rotation
    # is performed. At x=[0,0,0], there is a degeneracy that requires special handling
    # but this should not prevent the code from being used in optimization
    
    # Compute theta, no issues here
    theta = torch.norm(x)

    # Use the trick from `torch.nn.functional.normalize`, which adds a small epsilon to
    # the denominator to avoid division by zero
    v = torch.nn.functional.normalize(x, dim=0)

    # Apply the Rodrigues formula
    A = torch.zeros(3, 3, dtype=x.dtype, device=x.device)
    A[0,1], A[0,2], A[1,2] = -v[2], v[1], -v[0]
    K = A - A.T
    R = torch.eye(3, dtype=x.dtype, device=x.device) + torch.sin(theta) * K + (1-torch.cos(theta)) * (K @ K)
    return R


# Now we need to initialize the labeling of the sphere. We can try directly to use OMT to 
# match the sphere to each of the meshes and maybe that's going to be good enough for getting
# the initial label distributions. If not, have to deform
def to_measure(points, triangles):
    """Turns a triangle into a weighted point cloud."""

    # Our mesh is given as a collection of ABC triangles:
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    # Locations and weights of our Dirac atoms:
    X = (A + B + C) / 3  # centers of the faces
    S = torch.sqrt(torch.sum(torch.cross(B - A, C - A) ** 2, dim=1)) / 2  # areas of the faces

    # We return a (normalized) vector of weights + a "list" of points
    return S / torch.sum(S), X


# Compute optimal transport matching
def match_omt(vs, fs, vt, ft):
    """Match two triangle meshes using optimal mesh transport."""
    (a_src, x_src) = to_measure(vs, fs)
    (a_trg, x_trg) = to_measure(vt, ft)
    x_src.requires_grad_(True)
    x_trg.requires_grad_(True)

    # Generate correspondence between models using OMT
    t_start = time.time()
    w_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend='multiscale', verbose=True)
    w_loss_value = w_loss(a_src, x_src, a_trg, x_trg)
    [w_loss_grad] = torch.autograd.grad(w_loss_value, x_src)
    w_match = x_src - w_loss_grad / a_src[:, None]
    t_end = time.time()

    print(f'OMT matching distance: {w_loss_value.item()}, time elapsed: {t_end-t_start}')
    return w_loss_value, w_match


# Define a function that can fit a model to a population
def fit_model_to_population(md_root, md_targets, n_iter = 10, nt = 10,
                            sigma_lddmm=5, sigma_root=20, sigma_varifold=5, 
                            gamma_lddmm=0.1, w_jacobian_penalty=1.0):

    # LDDMM kernels
    device = md_root.vt.device
    K_root = GaussKernel(sigma=torch.tensor(sigma_root, dtype=torch.float32, device=device))
    K_temp = GaussKernel(sigma=torch.tensor(sigma_lddmm, dtype=torch.float32, device=device))
    K_vari = GaussLinKernelWithLabels(torch.tensor(sigma_varifold, dtype=torch.float32, device=device), md_root.lp.shape[1])

    # Create losses for each of the target meshes
    d_loss = { id: lossVarifoldSurfWithLabels(md_root.ft, v.vt, v.ft, md_root.lpt, v.lpt, K_vari) for id,v in md_targets.items() }
            
    # Create the root->template points/momentum, as well as the template->subject momenta
    q_root = torch.tensor(md_root.vt, dtype=torch.float32, device=device, requires_grad=True).contiguous()
    p_root = torch.zeros(md_root.vt.shape, dtype=torch.float32, device=device, requires_grad=True).contiguous()
    p_temp = torch.zeros((len(md_targets),) + md_root.vt.shape, dtype=torch.float32, device=device, requires_grad=True).contiguous()

    # Create the optimizer
    start = time.time()
    optimizer = torch.optim.LBFGS([p_root, p_temp], max_eval=16, max_iter=16, line_search_fn='strong_wolfe')
    n_subj = len(md_targets.items())

    def closure(detail = False):
        optimizer.zero_grad()

        # Shoot root->template
        _, q_temp = Shooting(p_root, q_root, K_root, nt)[-1]

        # Compute the triangle areas for the template
        z0 = q_temp[md_root.ft]
        area_0 = torch.norm(torch.cross(z0[:,1,:] - z0[:,0,:], z0[:,2,:] - z0[:,0,:]),dim=1) 

        # Make the momenta applied to the template average out to zero
        p_temp_z = p_temp - torch.mean(p_temp, 0, keepdim=True)

        # Compute the loss
        l_ham, l_data, l_jac = 0, 0, 0
        for i, (id,v) in enumerate(md_targets.items()):
            _, q_i = Shooting(p_temp_z[i,:], q_temp, K_temp, nt)[-1]

            z = q_i[md_root.ft]
            area = torch.norm(torch.cross(z[:,1,:] - z[:,0,:], z[:,2,:] - z[:,0,:]),dim=1)
            log_jac = torch.log10(area / area_0)

            l_ham = l_ham + gamma_lddmm * Hamiltonian(K_temp)(p_temp_z[i,:], q_temp)
            l_data = l_data + d_loss[id](q_i)
            l_jac = l_jac + torch.sum(log_jac ** 2) * w_jacobian_penalty

        l_ham, l_data, l_jac = l_ham / n_subj, l_data / n_subj, l_jac / n_subj
        L = l_ham + l_data + l_jac
        L.backward()

        # Return loss or detail
        if detail:
            return l_ham, l_data, l_jac, L
        else:
            return L

    # Perform optimization
    for i in range(n_iter):
        l_ham, l_data, l_jac, L = closure(True)
        print(f'Iteration {i:03d}  Loss H={l_ham:6.2f}  D={l_data:6.2f}  J={l_jac:6.2f}  Total={L:6.2f}')
        optimizer.step(closure)

    print(f'Optimization (L-BFGS) time: {round(time.time() - start, 2)} seconds')

    # Return the root model and the momenta
    p_temp_z = p_temp - torch.mean(p_temp, 0, keepdim=True)
    return p_root, p_temp_z 




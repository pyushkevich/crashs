# Kernel definitions (from KeOps)
import numpy as np
import torch
import time
import geomloss
from torch.autograd import grad
from vtkutil import *

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


# After OMT matching, we get a mapping from triangles to triangles. This code
# converts this into a vertex to vertex mapping, returning for every vertex in
# the fitted mesh a set of source vertices and their weights in the target mesh
def omt_match_to_vertex_weights(pd_fitted, pd_target, w_omt):
    pd_omt = vtk_clone_pd(pd_fitted)
    pd_omt = vtk_set_cell_array(pd_omt, 'match', w_omt)
    vtk_cell_array_to_point_array(pd_omt, 'match')
    v_omt = vtk_get_point_array(pd_omt, 'match')
    v_int, w_int = vtk_get_interpolation_arrays_for_sample(pd_target, v_omt)
    return v_omt, v_int, w_int
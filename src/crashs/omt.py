# Kernel definitions (from KeOps)
import numpy as np
import torch
import time
import geomloss
import SimpleITK as sitk
from torch.autograd import grad
from crashs.vtkutil import *

# Now we need to initialize the labeling of the sphere. We can try directly to use OMT to 
# match the sphere to each of the meshes and maybe that's going to be good enough for getting
# the initial label distributions. If not, have to deform
def to_measure(points, triangles):
    """Turns a triangle into a weighted point cloud."""

    # Our mesh is given as a collection of ABC triangles:
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    # Locations and weights of our Dirac atoms:
    X = (A + B + C) / 3  # centers of the faces
    S = torch.sqrt(torch.sum(torch.linalg.cross(B - A, C - A) ** 2, dim=1)) / 2  # areas of the faces

    # We return a (normalized) vector of weights + a "list" of points
    return S / torch.sum(S), X


# Compute optimal transport matching
def match_omt_old_deleteme(vs, fs, vt, ft):
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


# Match two measures using optimal mass transport, with normalization to diameter 1 sphere
def omt_match_measures(w_loss, a_src, x_src, a_trg, x_trg, normalize=True):
    """
    Match two measures using optimal mesh transport. 
    Returns Brenier map and OMT loss value    
    """
    
    # Normalize the measures to the sphere of diameter one. We don't need to worry
    # about the areas, since they are already normalized to one
    if normalize:
        ext_min, ext_max = torch.min(x_src, 0)[0], torch.max(x_src, 0)[0]
        scale = 1.0 / torch.max(ext_max - ext_min)
        shift = (ext_max + ext_min) / 2
    else:
        scale, shift = 1.0, 0.0
    
    xn_src, xn_trg = (x_src - shift) * scale, (x_trg - shift) * scale

    # Compute the approximate OMT plan
    xn_opt = xn_src.clone().detach().requires_grad_(True)
    w_loss_value = w_loss(a_src, xn_opt, a_trg, xn_trg)
    [w_loss_grad] = torch.autograd.grad(w_loss_value, xn_opt)
    xn_brenier = xn_src - w_loss_grad / a_src[:, None]

    # Undo the scaling
    return w_loss_value.item(), xn_brenier / scale + shift
 

def match_omt(vs, fs, vt, ft, normalize=True, **kwargs):
    """Match two triangle meshes using optimal mesh transport."""

    # Convert the meshes to measures 
    (a_src, x_src) = to_measure(vs, fs)
    (a_trg, x_trg) = to_measure(vt, ft)

    # Create a loss
    t_start = time.time()
    w_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, backend='multiscale', 
                                  verbose=False, scaling=0.8, 
                                  diameter=1.0 if normalize else None, 
                                  **kwargs)
    
    # Compute loss and Brienier map
    loss, w_match = omt_match_measures(w_loss, a_src, x_src, a_trg, x_trg, normalize=normalize)
    t_elapsed = time.time() - t_start

    # Compu
    print(f'OMT-match, {len(x_src)} -> {len(x_trg)}, L={loss:8.7f}, elapsed {t_elapsed:4.2f}')
    return loss, w_match


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


# Propagate a mesh through the levelset shells using OMT, starting with the 
# specified level. If no level specified, middle level will be used
def profile_meshing_omt_old_slow(img_levelset, device, source_mesh=None, init_layer=None, edge_len_pct=1.0):

    # Get the number of layers
    n = img_levelset.GetSize()[3]

    # The OMT algorithm that is called repeatedly
    def do_omt(l_src, l_trg):
        _, w_omt = match_omt_normalized(
            torch.tensor(vtk_get_points(l_src), dtype=torch.float32, device=device),
            torch.tensor(vtk_get_triangles(l_src), dtype=torch.int32, device=device),
            torch.tensor(vtk_get_points(l_trg), dtype=torch.float32, device=device),
            torch.tensor(vtk_get_triangles(l_trg), dtype=torch.int32, device=device))
        v_omt, _, _ = omt_match_to_vertex_weights(l_src, l_trg, w_omt.detach().cpu().numpy())
        return vtk_make_pd(v_omt, vtk_get_triangles(l_src))
    
    # Extract all the isolayers from the image 
    layers, matched = [], []
    for k in range(n):
        layers.append(extract_zero_levelset(img_levelset[:,:,:,k], edge_len_pct=edge_len_pct))
        matched.append(None)
    
    # Determine init layer
    if init_layer is None:
        init_layer = n // 2

    # Propagate from the mesh to the initial layer (unless we want to use a layer as source)
    if source_mesh:
        save_vtk(source_mesh, f'omt_init_src.vtk')
        save_vtk(layers[init_layer], f'omt_init_trg.vtk')
    matched[init_layer] = do_omt(source_mesh, layers[init_layer]) if source_mesh else layers[init_layer]

    # Propagate in both directions
    seq1 = reversed(range(0, init_layer))
    seq2 = reversed(range(n-1, init_layer, -1))
    for seq in seq1, seq2:
        prop_source = matched[init_layer]
        for k in seq:
            print(f'Propagating to layer {k}')
            save_vtk(prop_source, f'omt_{k}_src.vtk')
            save_vtk(layers[k], f'omt_{k}_trg.vtk')

            matched[k] = do_omt(prop_source, layers[k])
            prop_source = matched[k]

    # Return the propagated meshes
    return matched, init_layer



def profile_meshing_omt(img_levelset, device, source_mesh=None, init_layer=None, edge_len_pct=1.0):
    
    # An internal data structure for layers
    class Layer:
        def __init__(self, pd):
            self.pd = pd
            self.v = torch.tensor(vtk_get_points(pd), dtype=torch.float32, device=device)
            self.f = torch.tensor(vtk_get_triangles(pd), dtype=torch.int32, device=device)
            self.a, self.x = to_measure(self.v, self.f)
            self.f_match = None
            self.v_match = None

    # Get the number of layers
    n = img_levelset.GetSize()[3]

    # Extract all the isolayers from the image 
    layers = []
    for k in range(n):
        t_start = time.time()
        pd = extract_zero_levelset(img_levelset[:,:,:,k], edge_len_pct=edge_len_pct)
        layers.append(Layer(pd))
        t_elapsed = time.time() - t_start
        print(f'Extracting level set {k:02d}, elapsed: {t_elapsed:4.2f}')

    # Determine init layer
    if init_layer is None:
        init_layer = n // 2

    # Define the loss
    w_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, 
                                  backend='multiscale', verbose=False, 
                                  scaling=0.8, diameter=1.0)

    # Higher-level propagation command
    def flow(pd_flow, a_flow, x_flow, trg_layer):
        t_start = time.time()
        loss, trg_layer.f_match = omt_match_measures(
            w_loss, a_flow, x_flow, trg_layer.a, trg_layer.x, normalize=True)
        trg_layer.v_match, _, _ = omt_match_to_vertex_weights(
            pd_flow, trg_layer.pd, trg_layer.f_match.detach().cpu().numpy())
        t_elapsed = time.time() - t_start
        print(f'OMT-prop layer {k:02d}, {len(x_flow)} -> {len(trg_layer.x)}, L={loss:8.7f}, elapsed {t_elapsed:4.2f}')
        return trg_layer.f_match

    # If source layer is specified, then it is the first model that we must match
    if source_mesh:
        source_layer = Layer(source_mesh)
        pd_flow, a_flow = source_layer.pd, source_layer.a
        x_flow_init = flow(pd_flow, a_flow, source_layer.x, layers[init_layer])
    else:
        layers[init_layer].f_match = layers[init_layer].x
        layers[init_layer].v_match = layers[init_layer].v.detach().cpu().numpy()
        pd_flow, a_flow, x_flow_init = layers[init_layer].pd, layers[init_layer].a, layers[init_layer].x
    
    # Propagate in both directions
    seq1 = reversed(range(0, init_layer))
    seq2 = reversed(range(n-1, init_layer, -1))
    for seq in seq1, seq2:
        x_flow = x_flow_init
        for k in seq:
            x_flow = flow(pd_flow, a_flow, x_flow, layers[k])

    # If there is a source mesh, match it to target
    """
    if source_mesh:
        source_layer = Layer(source_mesh)
        target_layer = layers[init_layer]
        t_start = time.time()
        target_layer.f_match, loss = do_omt(source_layer.a, source_layer.x, target_layer.a, target_layer.x)
        target_layer.v_match, _, _ = omt_match_to_vertex_weights(
            source_layer.pd, target_layer.pd, target_layer.f_match.detach().cpu().numpy())
        print(np.min(target_layer.v_match), np.max(target_layer.v_match))
        t_elapsed = time.time() - t_start
        print(f'OMT-prop layer {init_layer:02d}, {len(source_layer.x)} -> {len(target_layer.x)}, L={loss:8.7f}, elapsed {t_elapsed:4.2f}')
        a_src, x_src = source_layer.a, target_layer.f_match
    else:
        source_layer = Layer(source_mesh)
        source_layer.f_match = source_layer.x
        source_layer.v_match = source_layer.v.detach().cpu().numpy()
        a_src, x_src = source_layer.a, source_layer.x

    # Propagate in both directions
    seq1 = reversed(range(0, init_layer))
    seq2 = reversed(range(n-1, init_layer, -1))
    for seq in seq1, seq2:
        x_flow = x_src
        for k in seq:
            t_start = time.time()
            layers[k].f_match, loss = do_omt(a_src, x_flow, layers[k].a, layers[k].x)
            layers[k].v_match, _, _ = omt_match_to_vertex_weights(
                layers[init_layer].pd, layers[k].pd, layers[k].f_match.detach().cpu().numpy())
            x_flow = layers[k].f_match
            t_elapsed = time.time() - t_start
            print(np.min(layers[k].v_match), np.max(layers[k].v_match))
            print(f'OMT-prop layer {k:02d}, {len(x_flow)} -> {len(layers[k].x)}, L={loss:8.7f}, elapsed {t_elapsed:4.2f}')
    """

    # Generate the meshes to return
    pd_result = []
    for i, layer in enumerate(layers):
        pd = vtk_make_pd(layer.v_match, source_layer.f.detach().cpu().numpy())
        vtk_set_cell_array(pd, 'wmatch', layer.f_match.detach().cpu().numpy())
        pd_result.append(pd)

    return pd_result, init_layer

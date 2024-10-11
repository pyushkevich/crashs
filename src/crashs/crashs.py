#!/usr/bin/env python3
import nighres
import torch
import SimpleITK as sitk
import numpy as np
import scipy
import os
import argparse
import json
import pathlib
import geomloss
import subprocess
import time
import glob
import SimpleITK as sitk
import tempfile
from picsl_c3d import Convert3D
from picsl_greedy import LMShoot3D
from picsl_cmrep import cmrep_vskel, mesh_tetra_sample
from crashs.util import *
from crashs.vtkutil import *
from crashs.lddmm import *
from crashs.omt import *
from crashs.preprocess_t2 import import_ashs_t2, add_wm_to_ashs_t1
from crashs.roi_integrate import integrate_over_rois

# Routine to convert ASHS posterior probabilities to CRUISE inputs
def ashs_output_to_cruise_input(template:Template, ashs: ASHSFolder, workspace: Workspace):
    
    # Convert posteriors to tissue probability images
    tissue_cat = ['wm', 'gm', 'bg']
    tissue_cat_labels = { k: template.get_labels_for_tissue_class(k) for k in tissue_cat }
    img_cat = ashs_posteriors_to_tissue_probabilities(ashs.posteriors, tissue_cat_labels, tissue_cat, 'bg')
    
    # Write each of the images
    def write(z, f_name):
        img_result = sitk.GetImageFromArray(z)
        img_result.CopyInformation(img_cat)
        sitk.WriteImage(img_result, f_name)

    # Write the probability images
    y = sitk.GetArrayFromImage(img_cat)
    write(y[:,:,:,0], workspace.cruise_wm_prob)
    write(y[:,:,:,1], workspace.cruise_gm_prob)
    write(y[:,:,:,2], workspace.cruise_bg_prob)

    # Threshold the white matter image
    b = np.where((y[:,:,:,0] - (y[:,:,:,1] + y[:,:,:,2])) > 0, 1, 0)

    # Take the largest connected components
    component_image = sitk.ConnectedComponent(sitk.GetImageFromArray(b), False)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    largest_component_binary_image.CopyInformation(img_cat)
    sitk.WriteImage(largest_component_binary_image, workspace.cruise_wm_mask)


# Routine to run CRUISE on an example
def run_cruise(workspace:Workspace, template:Template, overwrite=False):

    # These are the inputs to CRUISE
    cortex={
        'inside_mask': workspace.cruise_wm_mask,
        'inside_proba': workspace.cruise_wm_prob,
        'region_proba': workspace.cruise_gm_prob,
        'background_proba': workspace.cruise_bg_prob
        }

    # Append to fn_base for the CRUISE outputs (format consistent with earlier code)
    out_dir, fn_base = workspace.cruise_dir, workspace.cruise_fn_base

    # Correct the topology of the white matter segmentation
    corr = nighres.shape.topology_correction(
        image=cortex['inside_mask'],
        shape_type='binary_object',
        propagation='background->object',
        save_data=True,
        output_dir=out_dir,
        file_name=fn_base)

    # Use topology-preserving levelset to flow white matter to gray matter
    cruise = nighres.cortex.cruise_cortex_extraction(
        init_image=corr['object'],
        wm_image=cortex['inside_proba'],
        gm_image=cortex['region_proba'],
        csf_image=cortex['background_proba'],
        normalize_probabilities=True,
        save_data=True,
        file_name=fn_base,
        output_dir=out_dir)

    # Extract average surface, grey-white surface and grey-csf surface as meshes
    cortical_surface = {}
    for lset in ('avg', 'cgb', 'gwb'):
        cortical_surface[lset] = nighres.surface.levelset_to_mesh(
            levelset_image=cruise[lset],
            save_data=True,
            overwrite=overwrite,
            file_name=f'{fn_base}_{lset}.vtk',
            output_dir=out_dir)

    # Inflate the average surface 
    inflated_surface = nighres.surface.surface_inflation(
        surface_mesh=cortical_surface['avg']['result'],
        save_data=True,
        file_name=f'{fn_base}_avg.vtk',
        output_dir=out_dir, 
        overwrite=overwrite,
        step_size=0.1,
        max_iter=template.get_cruise_inflate_maxiter(), 
        method='area',
        regularization=1.0)

    # Compute cortical layering - this will be used later to extract meshes at different profiles
    depth = nighres.laminar.volumetric_layering(
                            inner_levelset=cruise['gwb'],
                            outer_levelset=cruise['cgb'],
                            n_layers=template.get_cruise_laminar_n_layers(),
                            method=template.get_cruise_laminar_method(),
                            save_data=True,
                            overwrite=overwrite,
                            file_name=fn_base,
                            output_dir=out_dir)

    # Generate corresponding surfaces at different layers
    """
    # No longer used - profile meshing happens using OMT after the template is fit
    profile_meshing = nighres.laminar.profile_meshing(
                            profile_surface_image=depth['boundaries'],
                            starting_surface_mesh=cortical_surface['avg']['result'],
                            save_data=True,
                            overwrite=overwrite,
                            file_name=f'{fn_base}.vtk',
                            output_dir=out_dir)
    """


# Helper function, get sform from ITK
def get_image_sform(img):
    dir = np.array(img.GetDirection()).reshape(3,3)
    spc = np.array(img.GetSpacing())
    off = np.array(img.GetOrigin())
    sform = np.eye(4)
    sform[:3,:3] = dir @ np.diag([spc[0], spc[1], spc[2]])
    sform[:3,3] = off
    sform = np.diag([-1, -1, 1, 1]) @ sform
    return sform


# CRUISE postprocessing command - takes meshes to RAS coordinate space
# and projects label probabilities onto the mesh
def cruise_postproc(template:Template, ashs:ASHSFolder, workspace: Workspace, reduction: None):

    # Get the matrix from voxel coordinates to RAS coordinates
    _, img_ref = next(iter(ashs.posteriors.items()))
    sform = get_image_sform(img_ref)

    # Save this matrix for future use
    np.savetxt(workspace.cruise_sform, sform)

    # Get the sampling grid from the image
    vox = sitk.GetArrayFromImage(img_ref)
    grid_spec = list(np.arange(vox.shape[k]) for k in range(3))

    # Load the inflated mesh - where the label probability arrays get saved
    pd_infl = load_vtk(workspace.cruise_infl_mesh)
    x_infl = vtk_get_points(pd_infl)

    # Load the midway mesh - should use it to sample
    pd_hw = load_vtk(workspace.cruise_middepth_mesh)
    x = vtk_get_points(pd_hw)

    # Sample the posterior at mesh coordinates
    labs = template.get_labels_surface_matching()
    plab = np.zeros((x_infl.shape[0], len(labs)))
    for i, label in enumerate(labs):
        img_itk = ashs.posteriors[label]
        if img_itk:
            vox = sitk.GetArrayFromImage(img_itk)
            plab[:,i] = scipy.interpolate.interpn(grid_spec, vox, np.flip(x, axis=1), bounds_error=False, fill_value=0.0)

    # Apply the sform to the mesh
    x_infl = x_infl @ sform[:3,:3].T + sform[:3,3:].T
    vtk_set_points(pd_infl, x_infl)

    # Softmax the labels
    plab = scipy.special.softmax(plab * 10., axis=1)    
    vtk_set_point_array(pd_infl, 'plab', plab)

    # Convert to a cell array
    pd_infl = vtk_all_point_arrays_to_cell_arrays(pd_infl) 

    # Save the mesh under a new name
    save_vtk(pd_infl, workspace.cruise_infl_mesh_labeled)

    # Also save the grey/white and grey/csf meshes from CRUISE in RAS space
    # This is only for visualization purposes
    for lset in ('cgb', 'gwb', 'avg'):
        pd_lset = load_vtk(workspace.fn_cruise(f'mtl_{lset}_l2m-mesh.vtk'))
        vtk_apply_sform(pd_lset, sform)
        save_vtk(pd_lset, workspace.fn_cruise(f'mtl_{lset}_l2m-mesh-ras.vtk'))

    # Apply the affine transform from ASHS, taking the mesh to template space
    M = np.linalg.inv(np.loadtxt(ashs.affine_to_template))
    x_infl = x_infl @ M[:3,:3].T + M[:3,3:].T 
    vtk_set_points(pd_infl, x_infl)
    save_vtk(pd_infl, workspace.affine_moving)

    # Downsample the inflated mesh. Since the inflated mesh is smooth, it should be
    # perfectly safe to downsample it by a large amount (say 10x), and still use it
    # as a target for varifold matching. 
    md_aff_reduced = MeshData(pd_infl, 'cpu', target_faces=reduction)
    md_aff_reduced.export(workspace.affine_moving_reduced)

    # Return the pd
    return pd_infl


# LDDMM registration between the template and the individual inflated mesh
# using lmshoot (much faster on the CPU than pykeops)
"""
def subject_to_template_registration_fast_cpu(template:Template, workspace: Workspace, device, reduction=None):

    # Load the template and put on the device
    fn_template_mesh = template.get_mesh(workspace.side)
    md_template = MeshData(load_vtk(fn_template_mesh), device, target_faces=reduction)
    md_template.export(workspace.fit_template_reduced)

    # Load the subject and put on the device
    md_subject = MeshData(load_vtk(workspace.affine_moving_reduced), device)

    # Call ml_affine for affine registration
    cmd = (f'ml_affine -m label {workspace.affine_moving_reduced} '
           f'{workspace.fit_template_reduced} {workspace.affine_matrix}')
    print('Executing: ', cmd)
    subprocess.run(cmd, shell=True)

    # Apply the affine registration
    maff = np.loadtxt(workspace.affine_matrix)
    pd_moving = load_vtk(workspace.affine_moving_reduced)
    x_moving = vtk_get_points(pd_moving)
    x_moving = x_moving @ maff[:3,:3].T + maff[:3,3:].T
    vtk_set_points(pd_moving, x_moving)
    save_vtk(pd_moving, workspace.fit_target_reduced)

    # Also apply the affine registration to the full model
    pd_moving_full = load_vtk(workspace.affine_moving)
    x_moving_full = vtk_get_points(pd_moving_full)
    x_moving_full = x_moving_full @ maff[:3,:3].T + maff[:3,3:].T
    vtk_set_points(pd_moving_full, x_moving_full)
    save_vtk(pd_moving_full, workspace.fit_target)

    # Parameter dictionary for the commands we are going to run
    cmd_param = {
        'template': workspace.fit_template_reduced,
        'template_full': fn_template_mesh,
        'target': workspace.fit_target_reduced,
        'fitted_full': workspace.fit_lddmm_result,
        'momenta': workspace.fit_lddmm_momenta_reduced,
        'sigma_lddmm': template.get_lddmm_sigma() / np.sqrt(2.0),
        'gamma': template.get_lddmm_gamma(),
        'steps': template.get_lddmm_nt(),
        'sigma_varifold': template.get_varifold_sigma() / np.sqrt(2.0)
    }

    # Run lmshoot to match to template
    cmd = ('lmshoot -d 3 -m {template} {target} -o {momenta} '
           '-s {sigma_lddmm} -l {gamma} -n {steps} -a V '
           '-S {sigma_varifold} -i 100 0 -t 1 -L plab').format(**cmd_param)
    print('Executing: ', cmd)
    subprocess.run(cmd, shell=True)

    # Run lmtowarp to apply the transformation to the full template
    cmd = ('lmtowarp -m {momenta} -n {steps} -d 3 -s {sigma_lddmm} '
           '-M {template_full} {fitted_full}').format(**cmd_param)
    print('Executing: ', cmd)
    subprocess.run(cmd, shell=True)
"""

# This data structure simplifies creation of temporary files from meshdata
class MeshDataTempFile:

    def __init__(self, md:MeshData, filename:str):
        _, self.fn_temp = tempfile.mkstemp(filename)
        md.export(self.fn_temp)

    def __str__(self):
        return self.fn_temp
    
    def __del__(self):
        pass
        ### os.remove(self.fn_temp)


# Perform similarity registration using external tools
def similarity_registration_lmshoot(md_temp, md_subj, fn_output, n_iter=50, sigma_varifold=10):

    # Save the meshes to temporary locations
    fn_temp = MeshDataTempFile(md_temp, 'template.vtk')
    fn_subj = MeshDataTempFile(md_subj, 'subject.vtk')

    # Run lmshoot to match to template
    lm = LMShoot3D()
    lm.fit(f'-m {fn_subj} {fn_temp} -o {fn_output} '
           f'-G -a V -S {sigma_varifold} -i {n_iter} 0 -t 1 -L plab')

    # Load the output matrix
    return np.loadtxt(fn_output)

    
# Affine registration between a subject and template, using varifold data term
# and Torch optimization
def similarity_registration_keops(md_temp, md_subj, n_iter=50, sigma_varifold=10):

    # Assign the inputs to a,b (easier to debug)
    m_a, m_b = md_subj, md_temp

    # LDDMM kernels
    device = md_temp.vt.device
    K_vari = GaussLinKernelWithLabels(torch.tensor(sigma_varifold, dtype=torch.float32, device=device), m_a.lp.shape[1])

    # Define the symmetric loss for this pair
    loss_ab = lossVarifoldSurfWithLabels(m_a.ft, m_b.vt, m_b.ft, m_a.lpt, m_b.lpt, K_vari)
    loss_ba = lossVarifoldSurfWithLabels(m_b.ft, m_a.vt, m_a.ft, m_b.lpt, m_a.lpt, K_vari)
    pair_theta = torch.tensor([0.01, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0], 
                              dtype=torch.float32, device=device, requires_grad=True)
    
    # Create optimizer
    opt_affine = torch.optim.LBFGS([pair_theta], max_eval=10, max_iter=10, line_search_fn='strong_wolfe')

    # Define closure
    def closure(detail=False):
        opt_affine.zero_grad()

        R = rotation_from_vector(pair_theta[0:3]) * pair_theta[3]
        b = pair_theta[4:]
        R_inv = torch.inverse(R)
        b_inv = - R_inv @ b

        a_to_b = (R @ m_a.vt.t()).t() + b
        b_to_a = (R_inv @ m_b.vt.t()).t() + b_inv

        L = 0.5 * (loss_ab(a_to_b) + loss_ba(b_to_a))
        L.backward()
        if not detail:
            return L
        else:
            return L, R, b
    
    # Run the optimization
    for i in range(n_iter):
        print(f'Affine Iteration {i:03d}  Loss: {closure().item()}')
        opt_affine.step(closure)

    # Return the loss and the transformation parameters
    loss, R, b = closure(True)
    affine_mat = np.eye(4)
    affine_mat[0:3,0:3] = R.detach().cpu().numpy()
    affine_mat[0:3,  3] = b.detach().cpu().numpy()

    # Print loss and record the best run/best parameters
    return loss.item(), affine_mat


def lddmm_fit_subject_jac_penalty_lmshoot(md_temp_full, md_temp, md_subj, 
                                          fn_momenta, fn_warped_full, fn_warped,
                                          n_iter=50, nt=10, sigma_lddmm=5, sigma_varifold=10, 
                                          gamma_lddmm=0.1, w_jac_penalty=1.0):
    
    # Save the meshes to temporary locations
    fn_temp_full = MeshDataTempFile(md_temp_full, 'template_full.vtk')
    fn_temp = MeshDataTempFile(md_temp, 'template.vtk')
    fn_subj = MeshDataTempFile(md_subj, 'subject.vtk')

    # Create a command to do registration
    lm = LMShoot3D()
    lm.fit(f'-m {fn_temp} {fn_subj} -o {fn_momenta} '
           f'-s {sigma_lddmm} -l 1 -g {gamma_lddmm} -J {w_jac_penalty} -R -n {nt} -a V '
           f'-S {sigma_varifold} -i {n_iter} 0 -t 1 -L plab')

    # Run lmtowarp to apply the transformation to the full template
    lm.apply(f'-m {fn_momenta} -R -n {nt} -d 3 -s {sigma_lddmm} '
             f'-M {fn_temp_full} {fn_warped_full} -M {fn_temp} {fn_warped}')


# Fit template to subject using LDDMM and KeOps, utilizing a Jacobian penalty
# if requested
def lddmm_fit_subject_jac_penalty_keops(md_temp, md_subj, n_iter=50, nt=10,
                                        sigma_lddmm=5, sigma_varifold=10, 
                                        gamma_lddmm=0.1, w_jac_penalty=1.0):

    # LDDMM kernels
    device = md_temp.vt.device
    K_temp = GaussKernel(sigma=torch.tensor(sigma_lddmm, dtype=torch.float32, device=device))
    K_vari = GaussLinKernelWithLabels(torch.tensor(sigma_varifold, dtype=torch.float32, device=device), md_temp.lp.shape[1])

    # Create losses for each of the target meshes
    d_loss = lossVarifoldSurfWithLabels(md_temp.ft, md_subj.vt, md_subj.ft, md_temp.lpt, md_subj.lpt, K_vari) 
            
    # Create the root->template points/momentum, as well as the template->subject momenta
    q_temp = md_temp.vt.clone().detach().requires_grad_(True).contiguous()
    p_temp = torch.zeros_like(q_temp).requires_grad_(True).contiguous()

    # Create the optimizer
    start = time.time()
    optimizer = torch.optim.LBFGS([p_temp], max_eval=16, max_iter=16, line_search_fn='strong_wolfe')

    z0 = md_temp.vt[md_temp.ft]
    area_0 = torch.norm(torch.cross(z0[:,1,:] - z0[:,0,:], z0[:,2,:] - z0[:,0,:]),dim=1) 

    def closure(detail = False):
        optimizer.zero_grad()
        _, q_i = Shooting(p_temp, q_temp, K_temp, nt)[-1]
        z = q_i[md_temp.ft]
        area = torch.norm(torch.cross(z[:,1,:] - z[:,0,:], z[:,2,:] - z[:,0,:]),dim=1)
        log_jac = torch.log10(area / area_0)

        l_ham = gamma_lddmm * Hamiltonian(K_temp)(p_temp, q_temp)
        l_data = d_loss(q_i)
        l_jac = torch.sum(log_jac ** 2) * w_jac_penalty
        L = l_ham + l_data + l_jac
        L.backward()
        if detail:
            return l_ham, l_data, l_jac, L
        else:
            return L

    # Perform optimization
    for i in range(n_iter):
        l_ham, l_data, l_jac, L = closure(True)
        print(f'Iteration {i:03d}  Losses: H={l_ham:8.6f}  D={l_data:8.6f}  J={l_jac:8.6f}  Total={L:8.6f}')
        optimizer.step(closure)

    print(f'Optimization (L-BFGS) time: {round(time.time() - start, 2)} seconds')

    # Apply shooting so we can return the fitted mesh too
    _, q_final = Shooting(p_temp, q_temp, K_temp, nt)[-1]

    # Return the root model and the momenta
    return p_temp, q_final


# LDDMM registration between the template and the individual inflated mesh
def subject_to_template_registration(template:Template, workspace: Workspace, device, 
                                     use_keops = None, reduction = None, lddmm_iter = None):
    
    # Determine whether to use keops if not specified
    if use_keops is None:
        use_keops = device.type == 'cuda'

    # Determine the number of iterations for affine and deformable
    affine_iter = template.get_affine_maxiter()
    lddmm_iter = lddmm_iter if lddmm_iter is not None else template.get_lddmm_maxiter()

    # Load the template and put on the device
    fn_template_mesh = template.get_mesh(workspace.side)
    md_template = MeshData(load_vtk(fn_template_mesh), device)

    # Load the subject and put on the device
    md_subject = MeshData(load_vtk(workspace.affine_moving), device)

    # Also load the reduced mesh
    md_subject_ds = MeshData(load_vtk(workspace.affine_moving_reduced), device)

    # Downsample the template for affine registration. Alternatively, the template may
    # provide its own downsampled mesh for fitting, in which case that's what we will use
    fn_template_mesh_reduced = template.get_reduced_mesh_for_lddmm(workspace.side)
    if fn_template_mesh_reduced:
        md_template_ds = MeshData(load_vtk(fn_template_mesh_reduced), device)
    else:
        md_template_ds = MeshData(vtk_clone_pd(md_template.pd), device, reduction)

    # Perform similarity registration
    print(f'Performing similarity and LDDMM registration')
    print(f'  template mesh: ({md_template_ds.v.shape[0]},{md_template_ds.f.shape[0]})')
    print(f'  target mesh:   ({md_subject_ds.v.shape[0]}, {md_subject_ds.f.shape[0]})')

    if use_keops:
        _, affine_mat = similarity_registration_keops(
            md_template_ds, md_subject_ds, n_iter=affine_iter, sigma_varifold=template.get_varifold_sigma())
            
        # Save the registration parameters
        np.savetxt(workspace.affine_matrix, affine_mat)

    else:
        affine_mat = similarity_registration_lmshoot(
            md_template_ds, md_subject_ds, workspace.affine_matrix, 
            n_iter=affine_iter, sigma_varifold=template.get_varifold_sigma())
    
    # Apply the affine registration parameters to the two meshes
    md_subject.apply_transform(affine_mat)
    md_subject.export(workspace.fit_target)

    md_subject_ds.apply_transform(affine_mat)
    md_subject_ds.export(workspace.fit_target_reduced)

    # Now perform the LDDMM deformation
    nt = template.get_lddmm_nt()
    if use_keops:
        # TODO: why is the target reduced and the template full-resolution?
        p_temp, q_fit = lddmm_fit_subject_jac_penalty_keops(
            md_template, md_subject_ds, n_iter=lddmm_iter, nt = nt,
            sigma_lddmm=template.get_lddmm_sigma(),
            sigma_varifold=template.get_varifold_sigma(),
            gamma_lddmm=template.get_lddmm_gamma(),
            w_jac_penalty=template.get_jacobian_penalty())
        
        # Save the fitting parameters
        pd = load_vtk(template.get_mesh(workspace.side))
        vtk_set_point_array(pd, 'Momentum', p_temp.cpu().detach().numpy())
        vtk_set_field_data(pd, 'lddmm_sigma', template.get_lddmm_sigma())
        vtk_set_field_data(pd, 'lddmm_nt', nt)
        vtk_set_field_data(pd, 'lddmm_ralston', 1.)
        save_vtk(pd, workspace.fit_lddmm_momenta)

        # We now need to combine the affine and deformable components to bring the mesh
        vtk_set_points(pd, q_fit.detach().cpu().numpy())
        save_vtk(pd, workspace.fit_lddmm_result)

    else:
        # Perform the fitting
        lddmm_fit_subject_jac_penalty_lmshoot(
            md_template, md_template_ds, md_subject_ds, 
            workspace.fit_lddmm_momenta, 
            workspace.fit_lddmm_result, workspace.fit_lddmm_result_reduced,
            n_iter=lddmm_iter, nt = nt,
            sigma_lddmm=template.get_lddmm_sigma(),
            sigma_varifold=template.get_varifold_sigma(),
            gamma_lddmm=template.get_lddmm_gamma(),
            w_jac_penalty=template.get_jacobian_penalty())
        
    # We now need to combine the affine and deformable components to bring the mesh
    # into the space of the subject
    #A_inv = np.linalg.inv(affine_mat[:3,:3])
    #b_inv = - A_inv @ affine_mat[:3,3:]


def subject_to_template_fit_omt(template:Template, workspace: Workspace, device):

    # Load the fitted template mesh
    pd_fitted = load_vtk(workspace.fit_lddmm_result)
    md_fitted = MeshData(pd_fitted, device)

    # Load the target subject mesh
    pd_subject = load_vtk(workspace.fit_target)
    md_subject = MeshData(pd_subject, device)

    # # Compute the centers and weights of the fitted model and target model
    # def to_measure(points, triangles):
    #     """Turns a triangle into a weighted point cloud."""

    #     # Our mesh is given as a collection of ABC triangles:
    #     A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    #     # Locations and weights of our Dirac atoms:
    #     X = (A + B + C) / 3  # centers of the faces
    #     S = torch.sqrt(torch.sum(torch.cross(B - A, C - A) ** 2, dim=1)) / 2  # areas of the faces

    #     # We return a (normalized) vector of weights + a "list" of points
    #     return S / torch.sum(S), X

    # # Compute optimal transport matching
    # (a_src, x_src) = to_measure(md_fitted.vt, md_fitted.ft)
    # (a_trg, x_trg) = to_measure(md_subject.vt, md_subject.ft)
    # x_src.requires_grad_(True)
    # x_trg.requires_grad_(True)

    # # Generate correspondence between models using OMT
    # t_start = time.time()
    # w_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend='multiscale', verbose=True)
    # w_loss_value = w_loss(a_src, x_src, a_trg, x_trg)
    # print('Forward pass completed')
    # [w_loss_grad] = torch.autograd.grad(w_loss_value, x_src)
    # w_match = x_src - w_loss_grad / a_src[:, None]
    # t_end = time.time()
    
    # print(f'OMT matching distance: {w_loss_value.item()}, time elapsed: {t_end-t_start}')
    
    # # The matches are the locations where the centers of the triangles want to move to
    # # on the target mesh. Now we need to map this into the corresponding point matches
    # pd_test_sinkhorn = vtk_clone_pd(pd_fitted)
    # vtk_set_cell_array(pd_test_sinkhorn, 'match', w_match.detach().cpu().numpy())
    # filter = vtk.vtkCellDataToPointData()
    # filter.SetInputData(pd_test_sinkhorn)
    # filter.Update()
    # vtk_set_points(pd_test_sinkhorn, vtk_get_point_array(filter.GetOutput(), 'match'))
    # save_vtk(pd_test_sinkhorn, workspace.fit_omt_match)

    # The last thing we want to do is to project template sampling locations into the
    # halfway surface in the subject native space
    pd_hw = load_vtk(workspace.cruise_middepth_mesh)

    # Apply RAS transform to the halfway mesh from CRUISE
    _, img_ref = next(iter(ashs.posteriors.items()))
    sform = get_image_sform(img_ref)
    x_hw = vtk_get_points(pd_hw) @ sform[:3,:3].T + sform[:3,3:].T
    vtk_set_points(pd_hw, x_hw)

    # Also load the label probability maps 
    plab_hw = vtk_get_cell_array(load_vtk(workspace.cruise_infl_mesh_labeled), 'plab')
    vtk_set_cell_array(pd_hw, 'plab', plab_hw)
    save_vtk(pd_hw, workspace.fit_omt_hw_target)

    # Which mesh to use for sampling
    pd_sample = pd_fitted 
    # pd_sample = pd_test_sinkhorn

    # Use the locator to sample from the halfway mesh
    loc = vtk.vtkCellLocator()
    loc.SetDataSet(pd_subject)
    loc.BuildLocator()
    x = vtk_get_points(pd_fitted)
    x_to_subj = np.zeros_like(x)
    x_dist = np.zeros(x.shape[0])
    
    cellId = vtk.reference(0)
    c = [0.0, 0.0, 0.0]
    subId = vtk.reference(0)
    d = vtk.reference(0.0)
    pcoord = [0.0, 0.0, 0.0]
    wgt = [0.0, 0.0, 0.0]
    xj = [0.0, 0.0, 0.0]
    for j in range(x.shape[0]):
        loc.FindClosestPoint(x[j,:], c, cellId, subId, d)
        pd_subject.GetCell(cellId).EvaluatePosition(x[j,:], c, subId, pcoord, d, wgt)
        pd_hw.GetCell(cellId).EvaluateLocation(subId, pcoord, xj, wgt)
        x_to_subj[j,:] = np.array(xj)
        x_dist[j] = np.sqrt(d.get())

    # Save the template locations in halfway mesh
    vtk_set_points(pd_fitted, x_to_subj)
    vtk_set_point_array(pd_fitted, 'dist', x_dist)
    save_vtk(pd_fitted, workspace.fit_omt_match_to_hw)

    # Compute distance statistics
    dist_stat = {
        'mean': np.mean(x_dist),
        'rms': np.sqrt(np.mean(x_dist ** 2)),
        'q95': np.quantile(x_dist, 0.95),
        'max': np.max(x_dist)
    }

    # Write distance statistics
    with open(workspace.fit_dist_stat, 'wt') as jsonfile:
        json.dump(dist_stat, jsonfile)


def omt_match_fitted_template_to_target(pd_fitted, pd_target, pd_target_native, device):

    # Load the fitted template mesh
    md_fitted, md_target = MeshData(pd_fitted, device), MeshData(pd_target, device)

    # Match the template to subject via OMT, i.e. every template vertex is mapped to somewhere on the
    # subject mesh, this fits more closely than LDDMM but might break topology
    _, w_omt = match_omt(md_fitted.vt, md_fitted.ft, md_target.vt, md_target.ft)
    v_omt, v_int, w_int = omt_match_to_vertex_weights(md_fitted.pd, md_target.pd, w_omt.detach().cpu().numpy())

    # Save the template OMT matched to the subject inflated mesh
    pd_omt = vtk_make_pd(v_omt, md_fitted.f)
    vtk_set_cell_array(pd_omt, 'plab', md_fitted.lp)
    vtk_set_cell_array(pd_omt, 'match', w_omt.detach().cpu().numpy())
    vtk_set_point_array(pd_omt, 'omt_v_int', v_int)
    vtk_set_point_array(pd_omt, 'omt_w_int', w_int)

    # Map the template to native mesh using the barycentric interpolation weights
    v_omt_to_native = np.einsum('vij,vi->vj', vtk_get_points(pd_target_native)[v_int,:], w_int)
    pd_omt_to_native = vtk_clone_pd(pd_omt)
    vtk_set_points(pd_omt_to_native, v_omt_to_native)

    # Return the two meshes
    return pd_omt, pd_omt_to_native


def subject_to_template_fit_omt_keops(template:Template, workspace: Workspace, device):

    # Perform matching, get the omt matches to halfway inflated mesh space and halfway native mesh
    pd_fitted = load_vtk(workspace.fit_lddmm_result)
    pd_subj_infl = load_vtk(workspace.fit_target)
    pd_subj_native = load_vtk(workspace.fn_cruise(f'mtl_avg_l2m-mesh.vtk'))
    pd_omt_to_infl, pd_omt_to_native = omt_match_fitted_template_to_target(pd_fitted, pd_subj_infl, pd_subj_native, device)

    # Save the template OMT matched to the subject inflated mesh
    save_vtk(pd_omt_to_infl, workspace.fit_omt_match)

    # Propagate this fitted mesh through the levelset layers using OMT
    print(f'Propagating through level set {id}')
    img_ls = sitk.ReadImage(workspace.fn_cruise('mtl_layering-boundaries.nii.gz'))
    prof_meshes, mid_layer = profile_meshing_omt(img_ls, source_mesh=pd_omt_to_native, device=device)

    # Get the sform for this subject
    sform = np.loadtxt(workspace.cruise_sform)

    # Save these profile meshes, but mapping to RAS coordinate space for compatibility
    # with image sampling tools
    vtk_apply_sform(pd_omt_to_native, sform)
    save_vtk(pd_omt_to_native, workspace.fit_omt_match_to_hw)
    for layer, pd_layer in enumerate(prof_meshes):
        vtk_apply_sform(pd_layer, sform, cell_coord_arrays=['wmatch'])
        save_vtk(pd_layer, workspace.fn_fit_profile_mesh(layer), binary=True)

    # Compute the distance statistics to establish quality of fit
    x_dist_ab = vtk_pointset_to_mesh_distance(pd_fitted, pd_subj_infl)
    x_dist_ba = vtk_pointset_to_mesh_distance(pd_subj_infl, pd_fitted)

    # Compute distance statistics
    dist_stat = {
        'mean': (np.mean(x_dist_ab) + np.mean(x_dist_ba)) / 2,
        'rms': (np.sqrt(np.mean(x_dist_ab ** 2)) + np.sqrt(np.mean(x_dist_ba ** 2))) / 2,
        'q95': (np.quantile(x_dist_ab, 0.95) + np.quantile(x_dist_ba, 0.95)) / 2,
        'max': np.maximum(np.max(x_dist_ab),np.max(x_dist_ba))
    }

    # Write distance statistics
    with open(workspace.fit_dist_stat, 'wt') as jsonfile:
        json.dump(dist_stat, jsonfile)


def compute_thickness_stats(template: Template, ws: Workspace):

    # Generate the levelset from which we will compute the thickness data
    fn_gwb = ws.fn_cruise('mtl_cruise-gwb.nii.gz')
    fn_cgb = ws.fn_cruise('mtl_cruise-cgb.nii.gz')
    c3d = Convert3D()
    c3d.execute(f'{fn_gwb} {fn_cgb} -scale -1 -min')
    img_levelset = c3d.peek(-1)

    # Extract a surface from this image
    pd_bnd = extract_zero_levelset(img_levelset)
    vtk_apply_sform(pd_bnd, get_image_sform(img_levelset))
    save_vtk(pd_bnd, ws.thick_boundary)

    # Apply a bit of smoothing to the mesh
    vs, fs = taubin_smooth(vtk_get_points(pd_bnd), vtk_get_triangles(pd_bnd), 0.5, -0.51, 40)
    pd_bnd_smooth = vtk_make_pd(vs, fs)
    save_vtk(pd_bnd_smooth, ws.thick_boundary_sm)

    # Compute the skeleton
    print('Calling cmrep_vskel with ')
    print(f'-e 2 -c 1 -p 1.6 -d {ws.thick_tetra_mesh} {ws.thick_boundary_sm} {ws.thick_skeleton}')
    cmrep_vskel(f'-e 2 -c 1 -p 1.6 -d {ws.thick_tetra_mesh} {ws.thick_boundary_sm} {ws.thick_skeleton}')

    # Sample the thickness from the tetrahedra onto the template grid
    print('Calling mesh_tetra_sample with ')
    print(f'-d 1.0 -B -D SamplingDistance {ws.fit_omt_match_to_hw} '
          f'{ws.thick_tetra_mesh} {ws.thick_result} VoronoiRadius')
    
    mesh_tetra_sample(f'-d 1.0 -B -D SamplingDistance {ws.fit_omt_match_to_hw} '
                      f'{ws.thick_tetra_mesh} {ws.thick_result} VoronoiRadius')
    
    # Compute thickness summary measures
    integrate_over_rois(template, argparse.Namespace(
        subject=ws.expid, session=None, scan=None, side=ws.side,
        array=['VoronoiRadius'], mesh=ws.thick_result, output=ws.thick_roi_summary))


# The main program launcher
class FitLauncher:

    def __init__(self, parse):

        # Add the arguments
        parse.add_argument('ashs_dir', metavar='ashs_dir', type=pathlib.Path, 
                        help='ASHS output directory')
        parse.add_argument('template', metavar='template', type=str, 
                        help='Name of the CRASHS template (folder in $CRASHS_DATA/templates)')
        parse.add_argument('output_dir', metavar='output_dir', type=str, 
                        help='Output directory to save images')
        parse.add_argument('-C', '--crashs-data', metavar='dir', type=str,
                           help='Path of the CRASHS data folder, if CRASHS_DATA not set')
        parse.add_argument('-i', '--id', metavar='id', type=str, 
                        help='Experiment id, defaults to output directory basename')
        parse.add_argument('-s', '--side', type=str, choices=['left', 'right'], 
                        help='Side of the brain', default='left')
        parse.add_argument('-f', '--fusion-stage', type=str, choices=['multiatlas', 'bootstrap'], 
                        help='Which stage of ASHS fusion to select', default='bootstrap')                   
        parse.add_argument('-c', '--correction-mode', type=str, choices=['heur', 'corr_usegray', 'corr_nogray'], 
                        help='Which ASHS correction output to select', default='corr_usegray')                   
        parse.add_argument('-d', '--device', type=str, 
                        help='PyTorch device to use (cpu, cuda0, etc)', default='cpu')
        parse.add_argument('-K', '--keops', action='store_true',
                        help='Use KeOps routines for registration and OMT (GPU needed)')
        parse.add_argument('-r', '--reduction', type=float, 
                        help='Reduction to apply to meshes for geodesic shooting', default=None)
        parse.add_argument('--lddmm-iter', type=int, default=None,
                        help='Number of iterations for geodesic shooting')
        parse.add_argument('--skip-preproc', action='store_true',
                        help='Skip the preprocessing step')
        parse.add_argument('--skip-cruise', action='store_true',
                        help='Skip the surface reconstruction step')
        parse.add_argument('--skip-reg', action='store_true',
                        help='Skip the registration step')
        parse.add_argument('--skip-omt', action='store_true',
                        help='Skip the optimal transport matching step')
        parse.add_argument('--skip-thick', action='store_true',
                        help='Skip the thickness computation step')

        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))


    def run(self, args):

        # Load the template
        cdr = CrashsDataRoot(args.crashs_data)
        template = Template(cdr.find_template(args.template))

        # Some parameters can be specified in the template or by user
        reduction = template.get_cruise_inflate_reduction() 
        if args.reduction is not None:
            reduction = args.reduction

        # Load the ASHS experiment
        ashs = ASHSFolder(args.ashs_dir, args.side, args.fusion_stage, args.correction_mode)
        ashs.load_posteriors(template)

        # Create the output workspace
        expid = args.id if args.id is not None else os.path.basename(args.output_dir)
        workspace = Workspace(args.output_dir, expid, args.side)

        # Check if the white matter is present in ASHS
        have_wm = len([l for l in template.get_labels_for_tissue_class('wm') 
                       if l in ashs.posteriors.keys()]) > 0

        # Determine the device to use in torch
        device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
        print("CUDA available    : ", torch.cuda.is_available())
        print("CUDA device count : ", torch.cuda.device_count())
        print("Selected device   : ", device)

        # Determine if the current template requires additional postprocessing steps
        # to convert the ASHS output into an input suitable for CRASHS
        if args.skip_preproc is False:
            if template.get_preprocessing_mode() == 't2_alveus':
                fn_preproc = f'{args.output_dir}/preprocess/t2_alveus'
                print("Performing ASHS-T2 preprocessing steps (alveus WM wrap)")
                upsampled_posterior_pattern = import_ashs_t2(cdr, ashs, template, fn_preproc, expid, device)
                ashs.set_alternate_posteriors(upsampled_posterior_pattern)
                ashs.load_posteriors(template)
            elif not have_wm and template.get_preprocessing_mode() == 't1_add_wm':
                fn_preproc = f'{args.output_dir}/preprocess/t1_add_wm'
                print("Performing ASHS-T1 preprocessing steps (add WM label)")
                upsampled_posterior_pattern = add_wm_to_ashs_t1(cdr, ashs, template, fn_preproc, expid, device)
                ashs.set_alternate_posteriors(upsampled_posterior_pattern)
                ashs.load_posteriors(template)
            elif not have_wm:
                raise ValueError('ASHS folder missing white matter segmentation!')

        # Perform the import and CRUISE steps
        if args.skip_cruise is False:

            # Convert the inputs into probability maps
            print("Converting ASHS-T1 posteriors to CRUISE inputs")
            ashs_output_to_cruise_input(template, ashs, workspace)

            # Run CRUISE on the inputs
            print("Running CRUISE to correct topology and compute inflated surface")
            run_cruise(workspace, template, overwrite=True)

            # Perform postprocessing
            print("Mapping ASHS labels on the inflated template")
            cruise_postproc(template, ashs, workspace, reduction=reduction)

        # Perform the registration
        if args.skip_reg is False:

            # Affine and LDDMM registration between template and subject
            print("Registering template to subject")
            subject_to_template_registration(template, workspace, device, 
                                             use_keops=args.keops, reduction=reduction, lddmm_iter=args.lddmm_iter)

        if args.skip_omt is False:
            # Perform the OMT matching
            print("Matching template to subject using OMT")
            subject_to_template_fit_omt_keops(template, workspace, device)

        # Compute thickness statistics
        if args.skip_thick is False:
            compute_thickness_stats(template, workspace)


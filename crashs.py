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
from vtkutil import *
from lddmm import *

# Class that represents a surface mesh for use in PyTorch
class MeshData:
    def __init__(self, pd:vtk.vtkPolyData, device, target_faces=None):
        self.pd = vtk_all_point_arrays_to_cell_arrays(pd)
        self.v, self.f = vtk_get_points(self.pd), vtk_get_triangles(self.pd)
        self.lp = vtk_get_cell_array(self.pd, 'plab')
        if target_faces:
            vd, fd = decimate(self.v, self.f, target_faces)
            self.lp = vtk_sample_cell_array_at_vertices(self.pd, self.lp, vd)
            self.v, self.f = vd, fd
        self.vt = torch.tensor(self.v, dtype=torch.float32, device=device).contiguous()
        self.ft = torch.tensor(self.f, dtype=torch.long, device=device).contiguous()
        self.lpt = torch.tensor(self.lp, dtype=torch.float32, device=device).contiguous()


# Class that represents information about a template
class Template :
    def __init__(self, template_dir, side):
        # Load the template json
        with open(os.path.join(template_dir, 'template.json')) as template_json:
            self.json = json.load(template_json)

        # Load the template mesh
        self.pd = load_vtk(os.path.join(template_dir, self.json['sides'][side]['mesh']))

    def get_labels_for_tissue_class(self, tissue_class):
        return self.json['ashs_label_type'][tissue_class]
    
    def get_labels_surface_matching(self):
        return self.json['labels_for_surface_matching']

    def get_varifold_sigma(self):
        return self.json['registration']['varifold_sigma']

    def get_lddmm_sigma(self):
        return self.json['registration']['lddmm_sigma']

    def get_lddmm_gamma(self):
        return self.json['registration']['lddmm_gamma']

    def get_lddmm_nt(self):
        return self.json['registration']['lddmm_nt']


        
# Class to represent an ASHS output folder and the files that we need from
# ASHS for this script
class ASHSFolder:
    def __init__(self, ashs_dir, side, fusion_mode, correction_mode):

        # How posteriors are coded
        pstr = 'posterior' if correction_mode == 'heur' else f'posterior_{correction_mode}'

        # Set the posterior pattern
        self.posterior_pattern = os.path.join(
            ashs_dir, 
            f'{fusion_mode}/fusion/{pstr}_{side}_%03d.nii.gz')

        # Load the affine transform to the template
        self.affine_to_template = np.loadtxt(os.path.join(
            ashs_dir, 'affine_t1_to_template/t1_to_template_affine.mat'))
        

    def load_posteriors(self, template:Template):

        # Load the posteriors
        self.posteriors = {}
        for lab in ['wm', 'gm', 'bg']:
            for v in template.get_labels_for_tissue_class(lab):
                img=self.posterior_pattern % (v,)
                if os.path.exists(img):
                    self.posteriors[v] = sitk.ReadImage(img)


class Workspace:

    # Define output files
    def fn_cruise(self, suffix):
        return os.path.join(self.cruise_dir, f'{self.expid}_{suffix}')

    # Define output files
    def fn_fit(self, suffix):
        return os.path.join(self.fit_dir, f'{self.expid}_{suffix}')

    def __init__(self, output_dir, expid):
        self.output_dir = output_dir
        self.expid = expid

        # Define and create output directories
        self.cruise_dir = os.path.join(self.output_dir, 'cruise')
        self.fit_dir = os.path.join(self.output_dir, 'fitting')
        os.makedirs(self.cruise_dir, exist_ok=True)
        os.makedirs(self.fit_dir, exist_ok=True)

        self.cruise_fn_base = f'{self.expid}_mtl'
        self.cruise_wm_prob = self.fn_cruise('wm_prob.nii.gz')
        self.cruise_gm_prob = self.fn_cruise('gm_prob.nii.gz')
        self.cruise_bg_prob = self.fn_cruise('bg_prob.nii.gz')
        self.cruise_wm_mask = self.fn_cruise('wm_mask_lcomp.nii.gz')
        self.cruise_infl_mesh = self.fn_cruise('mtl_avg_infl-mesh.vtk')
        self.cruise_infl_mesh_labeled = self.fn_cruise('mtl_avg_infl-mesh-ras-labeled.vtk')
        self.cruise_middepth_mesh = self.fn_cruise('mtl_mesh-p2.vtk')

        self.affine_moving = self.fn_fit('affine_moving.vtk')
        self.affine_moving_reduced = self.fn_fit('affine_moving_reduced.vtk')
        self.affine_matrix = self.fn_fit('affine_matrix.mat')
        self.fit_template = self.fn_fit('fit_template.vtk')
        self.fit_target = self.fn_fit('fit_target.vtk')
        self.fit_target_reduced = self.fn_fit('fit_target_reduced.vtk')
        self.fit_template_reduced = self.fn_fit('fit_template_reduced.vtk')
        self.fit_lddmm_momenta_reduced = self.fn_fit('fit_lddmm_momenta_reduced.vtk')
        self.fit_lddmm_momenta = self.fn_fit('fit_lddmm_momenta.vtk')
        self.fit_lddmm_result = self.fn_fit('fitted_lddmm_template.vtk')
        self.fit_omt_match = self.fn_fit('fitted_omt_match.vtk')
        self.fit_omt_hw_target = self.fn_fit('fitted_omt_hw_target.vtk')
        self.fit_omt_match_to_hw = self.fn_fit('fitted_omt_match_to_hw.vtk')
        self.fit_dist_stat = self.fn_fit('fitted_dist_stat.json')



# Load the available label posteriors into a dictionary
#def load_ashs_posteriors(template:Template, posterior_pattern):
#    p = {}
#    for lab in ['wm', 'gm', 'bg']:
#        for v in template.get_labels_for_tissue_class(lab):
#            img=posterior_pattern % (v,)
#            if os.path.exists(img):
#                p[v] = sitk.ReadImage(img)
#    return p


# Routine to convert ASHS posterior probabilities to CRUISE inputs
def ashs_output_to_cruise_input(template:Template, ashs_posteriors: dict, workspace: Workspace):
    
    # Available posterior images for each class
    idat = {}

    # Load the posteriors if they exist and if not, fill in with a zero image
    for lab in ['wm', 'gm', 'bg']:
        idat[lab] = {}
        prob_max=None
        for v in template.get_labels_for_tissue_class(lab):
            img_itk = ashs_posteriors.get(v, None)
            if img_itk:
                img_arr = sitk.GetArrayFromImage(img_itk)
                if prob_max is None:
                    prob_max = img_arr
                    idat[lab]['itk'] = img_itk
                else:
                    prob_max = np.maximum(prob_max, img_arr)

        idat[lab]['prob_max'] = prob_max

    # Create a single numpy array combining the probabilities
    x = np.concatenate((idat['wm']['prob_max'][:, :, :, np.newaxis],
                        idat['gm']['prob_max'][:, :, :, np.newaxis],
                        idat['bg']['prob_max'][:, :, :, np.newaxis]), axis=3)

    # Replace missing data with background probability
    x[:,:,:,2] = np.where(x.max(axis=3) == 0., 1.0, x[:,:,:,2])

    # Compute the softmax
    y = scipy.special.softmax(x * 10, axis=3)

    # Write each of the images
    def write(z, f_name):
        img_result = sitk.GetImageFromArray(z)
        img_result.CopyInformation(idat['wm']['itk'])
        sitk.WriteImage(img_result, f_name)

    # Write the probability images
    write(y[:,:,:,0], workspace.cruise_wm_prob)
    write(y[:,:,:,1], workspace.cruise_gm_prob)
    write(y[:,:,:,2], workspace.cruise_bg_prob)

    # Threshold the white matter image
    b = np.where((y[:,:,:,0] - (y[:,:,:,1] + y[:,:,:,2])) > 0, 1, 0)

    # Take the largest connected components
    component_image = sitk.ConnectedComponent(sitk.GetImageFromArray(b), False)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    largest_component_binary_image.CopyInformation(idat['wm']['itk'])
    sitk.WriteImage(largest_component_binary_image, workspace.cruise_wm_mask)


# Routine to run CRUISE on an example
def run_cruise(workspace:Workspace, overwrite=False):
  
  # These are the inputs to CRUISE
  cortex={
    'inside_mask': workspace.cruise_wm_mask,
    'inside_proba': workspace.cruise_wm_prob,
    'region_proba': workspace.cruise_gm_prob,
    'background_proba': workspace.cruise_bg_prob
  }

  # Append to fn_base for the CRUISE outputs (format consistent with earlier code)
  out_dir, fn_base = workspace.cruise_dir, workspace.cruise_fn_base

  # Get the directory and filename base for the CRUISE commands
  corr=nighres.shape.topology_correction(
    image=cortex['inside_mask'],
    shape_type='binary_object',
    propagation='background->object',
    save_data=True,
    output_dir=out_dir,
    file_name=fn_base)
	
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
    # Extract the level set
    cortical_surface[lset] = nighres.surface.levelset_to_mesh(
        levelset_image=cruise[lset],
        save_data=True,
        overwrite=overwrite,
        file_name=f'{fn_base}_{lset}.vtk',
        output_dir=out_dir)    

  inflated_surface = nighres.surface.surface_inflation(
                          surface_mesh=cortical_surface['avg']['result'],
                          save_data=True,
                          file_name=f'{fn_base}_avg.vtk',
                          output_dir=out_dir, 
                          overwrite=overwrite,
                          step_size=0.1,
                          max_iter=500, 
                          method='area',
                          regularization=1.0)

  depth = nighres.laminar.volumetric_layering(
                          inner_levelset=cruise['gwb'],
                          outer_levelset=cruise['cgb'],
                          n_layers=4,
                          method='distance-preserving',
                          save_data=True,
                          overwrite=overwrite,
                          file_name=fn_base,
                          output_dir=out_dir)

  # Generate corresponding surfaces at different layers
  profile_meshing = nighres.laminar.profile_meshing(
                          profile_surface_image=depth['boundaries'],
                          starting_surface_mesh=cortical_surface['avg']['result'],
                          save_data=True,
                          overwrite=overwrite,
                          file_name=f'{fn_base}.vtk',
                          output_dir=out_dir)


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


# Compute label probability maps on the mesh for relevant labels
def mesh_label_prob_maps(pd):
    posteriors = { 0: 20, 1: 1, 2: 2, 3: 10, 4: 11, 5: 12, 6: 13 }
    
    # Load all the maps into a single array
    d = np.zeros((pd.GetNumberOfCells(), 21))
    for l in range(21):
        a = vtk_get_cell_array(pd, f'label_{l:03d}')
        if a is not None:
            d[:,l] = a
            
    # Combine the labels we care about
    q = np.zeros((pd.GetNumberOfCells(), 7))
    for i,m in posteriors.items():
        q[:,i] = d[:,m]
    
    # Compute softmax
    return scipy.special.softmax(q * 10., axis=1)


# CRUISE postprocessing command - takes meshes to RAS coordinate space
# and projects label probabilities onto the mesh
def cruise_postproc(template:Template, ashs:ASHSFolder, workspace: Workspace):

    # Get the matrix from voxel coordinates to RAS coordinates
    _, img_ref = next(iter(ashs.posteriors.items()))
    sform = get_image_sform(img_ref)
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
    for lset in ('cgb', 'gwb'):
        pd_lset = load_vtk(workspace.fn_cruise(f'mtl_{lset}_l2m-mesh.vtk'))
        x_lset = vtk_get_points(pd_lset)
        x_lset = x_lset @ sform[:3,:3].T + sform[:3,3:].T
        vtk_set_points(pd_lset, x_lset)
        save_vtk(pd_lset, workspace.fn_cruise(f'mtl_{lset}_l2m-mesh-ras.vtk'))

    # Apply the affine transform from ASHS, taking the mesh to template space
    M = np.linalg.inv(ashs.affine_to_template)
    x_infl = x_infl @ M[:3,:3].T + M[:3,3:].T 
    vtk_set_points(pd_infl, x_infl)
    save_vtk(pd_infl, workspace.affine_moving)

    # Save a local copy of the template with the label array as a point
    # array (which matches the target)
    save_vtk(template.pd, workspace.fit_template)

    # Return the pd
    return pd_infl


# LDDMM registration between the template and the individual inflated mesh
# using lmshoot (much faster on the CPU than pykeops)
def subject_to_template_registration_fast_cpu(template:Template, workspace: Workspace, device, reduction=None):

    # Load the template and put on the device
    md_template = MeshData(template.pd, device, reduction)

    # Load the subject and put on the device
    md_subject = MeshData(load_vtk(workspace.affine_moving), device, reduction)

    # Export the two reduced meshes
    def export_reduced(md, fn):
        pd_reduced = vtk_make_pd(md.v, md.f)
        pd_reduced = vtk_set_point_array(pd_reduced, 'plab', md.lp)
        pd_reduced = vtk_all_point_arrays_to_cell_arrays(pd_reduced)
        vtk_set_point_array(pd_reduced, 'label', np.argmax(md.lp, axis=1))
        save_vtk(pd_reduced, fn)

    export_reduced(md_template, workspace.fit_template_reduced) 
    export_reduced(md_subject, workspace.affine_moving_reduced) 

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
        'template_full': workspace.fit_template,
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


# LDDMM registration between the template and the individual inflated mesh
def subject_to_template_registration(template:Template, workspace: Workspace, device, reduction=None):

    # Load the template and put on the device
    md_template = MeshData(template.pd, device, reduction)

    # Load the subject and put on the device
    pd_subj = load_vtk(workspace.fit_target)
    md_subject = MeshData(pd_subj, device, reduction)

    # Set the sigma tensors
    sigma_varifold = torch.tensor(
        [template.get_varifold_sigma()], dtype=md_template.vt.dtype, device=device)
    sigma_lddmm = torch.tensor(
        [template.get_lddmm_sigma()], dtype=md_template.vt.dtype, device=device)

    # Data loss with label similarity, target here is the template
    dloss = lossVarifoldSurfWithLabels(
        md_template.ft, md_subject.vt, md_subject.ft,
        md_template.lpt, md_subject.lpt, 
        GaussLinKernelWithLabels(sigma_varifold, md_template.lp.shape[1]))

    # Overall LDDMM loss
    nt = template.get_lddmm_nt()
    loss = LDDMMloss(
        GaussKernel(sigma=sigma_lddmm), dloss, nt, template.get_lddmm_gamma())

    # Define the momenta
    q = md_template.vt.clone().requires_grad_(True).contiguous()
    p = torch.zeros_like(md_template.vt).requires_grad_(True).contiguous()

    # Create an optimizer
    optimizer = torch.optim.LBFGS([p], max_eval=20, max_iter=20, line_search_fn='strong_wolfe')

    # Define a rigid transformation using the Rodrigues formula
    def closure():
        optimizer.zero_grad()
        L = loss(p, q)
        L.backward()
        print(f'Loss: {L.item()}')
        return L
    
    for i in range(20):
        optimizer.step(closure)

    # Save the fitting parameters
    pd = vtk_clone_pd(template.pd)
    vtk_set_point_array(pd, 'Momentum', p.cpu().detach().numpy())
    vtk_set_field_data(pd, 'lddmm_sigma', template.get_lddmm_sigma())
    vtk_set_field_data(pd, 'lddmm_nt', nt)
    save_vtk(pd, workspace.fit_lddmm_momenta)

    # Save the mesh under a new name
    _, q_fit=Shooting(p, q, GaussKernel(sigma=sigma_lddmm), nt)[-1]
    vtk_set_points(pd, q_fit.cpu().detach().numpy())
    save_vtk(pd, workspace.fit_lddmm_result)


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



# Create a parser
parse = argparse.ArgumentParser(description="ASHS-T1 Surface-Based Analysis based on CRUISE")

# Add the arguments
parse.add_argument('ashs_dir', metavar='ASHS output path', type=pathlib.Path, 
                   help='Directory generated by running T1-ASHS')
parse.add_argument('template_dir', metavar='CrASHS tempate path', type=pathlib.Path, 
                   help='Path to the CrASHS template')
parse.add_argument('output_dir', metavar='output_dir', type=str, 
                   help='Output directory to save images')
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
parse.add_argument('-r', '--reduction', type=float, 
                   help='Reduction to apply to meshes for geodesic shooting', default=None)
args = parse.parse_args()

# Load the template
template = Template(args.template_dir, args.side)

# Load the ASHS experiment
ashs = ASHSFolder(args.ashs_dir, args.side, args.fusion_stage, args.correction_mode)
ashs.load_posteriors(template)

# Create the output workspace
expid = args.id if args.id is not None else os.path.basename(args.output_dir)
workspace = Workspace(args.output_dir, expid)

# Determine the device to use in torch
device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'

# Convert the inputs into probability maps
print("Converting ASHS-T1 posteriors to CRUISE inputs")
ashs_output_to_cruise_input(template, ashs.posteriors, workspace)

# Run CRUISE on the inputs
print("Running CRUISE to correct topology and compute inflated surface")
run_cruise(workspace, overwrite=True)

# Perform postprocessing
print("Mapping ASHS labels on the inflated template")
cruise_postproc(template, ashs, workspace)

# Perform the registration
print("Registering template to subject")
subject_to_template_registration_fast_cpu(template, workspace, device, args.reduction)

# Perform the OMT matching
print("Matching template to subject using OMT")
subject_to_template_fit_omt(template, workspace, device)


import argparse
import pathlib
import json
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.decomposition import PCA
from crashs.vtkutil import *
from crashs.lddmm import *
from crashs.omt import *
from crashs.util import MeshData, Template, ASHSFolder, Workspace
from crashs.crashs import ashs_output_to_cruise_input, run_cruise, cruise_postproc
from crashs.crashs import omt_match_fitted_template_to_target

# Workspace for building the template
class TemplateBuildWorkspace:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._subj = {}

    def add_subject(self, id, workspace:Workspace, side):
        self._subj[id] = {'side': side, 'workspace': workspace}

    def get_subjects(self):
        return self._subj.items()
    
    def get_ids(self):
        return list(self._subj.keys())
    
    def get_subject_workspace(self, id):
        return self._subj[id]['workspace']

    def subj_dir(self, id):
        dir = os.path.join(self.output_dir, id)
        os.makedirs(dir, exist_ok=True)
        return dir
    
    def affine_dir(self):
        dir = os.path.join(self.output_dir, f'groupwise_affine')
        os.makedirs(dir, exist_ok=True)
        return dir

    def lddmm_dir(self):
        dir = os.path.join(self.output_dir, f'groupwise_lddmm')
        os.makedirs(dir, exist_ok=True)
        return dir

    def final_dir(self):
        dir = os.path.join(self.output_dir, f'groupwise_final')
        os.makedirs(dir, exist_ok=True)
        return dir

    def fn_affine_input(self, id:str):
        return os.path.join(self.affine_dir(), f'affine_input_{id}.vtk')

    def fn_affine_mesh(self, id:str):
        return os.path.join(self.affine_dir(), f'affine_aligned_{id}.vtk')

    def fn_affine_mesh_reduced(self, id:str):
        return os.path.join(self.affine_dir(), f'affine_aligned_reduced_{id}.vtk')

    def fn_affine_matrix(self, id:str):
        return os.path.join(self.affine_dir(), f'affine_{id}.mat')

    def fn_template_build_mesh(self, iter:int):
        return os.path.join(self.iter_dir(iter), f'template_iter_{iter}.vtk')
    
    def fn_template_build_tosubj(self, iter:int, id:str):
        return os.path.join(self.iter_dir(iter), f'template_{iter}_to_{id}.vtk')
    
    def fn_final_fitted_mesh(self, id:str):
        return os.path.join(self.final_dir(), f'template_fitted_{id}.vtk')
    
    def fn_final_omt_mesh(self, id:str):
        return os.path.join(self.final_dir(), f'template_omt_{id}.vtk')
    
    def fn_final_omt_to_native_hw_mesh(self, id:str):
        return os.path.join(self.final_dir(), f'template_omt_to_native_hw_{id}.vtk')
    
    def qa_dir(self):
        dir = os.path.join(self.output_dir, f'qa')
        os.makedirs(dir, exist_ok=True)
        return dir
    

# Run the groupwise affine step and find the mesh to use as the template source
def groupwise_similarity_registration_keops(tbs: TemplateBuildWorkspace, template: Template, device):

    # From the template directory, load the left/right flip file. 
    flip_lr = np.loadtxt(os.path.join(template.root, 'ashs_template_flip.mat'))

    # Set the sigma tensors
    sigma_varifold = torch.tensor([template.get_varifold_sigma()], dtype=torch.float32, device=device)

    # Load each of the meshes that will be used to build the template
    md_full, md_ds = {}, {}
    for id, sd in tbs.get_subjects():
        
        # Depending on the side, apply flip_lr as the transform
        transform = flip_lr if sd['side'] == 'right' else None

        # Load the mesh data
        pd = load_vtk(sd['workspace'].affine_moving)
        md_full[id] = MeshData(load_vtk(sd['workspace'].affine_moving), device, transform=transform)
        md_ds[id] = MeshData(load_vtk(sd['workspace'].affine_moving_reduced), device, transform=transform)

        # Save the input to the groupwise affine
        pd_flipped = vtk_make_pd(md_ds[id].v, md_ds[id].f)
        vtk_set_cell_array(pd_flipped, 'plab', md_ds[id].lp)
        save_vtk(pd_flipped, tbs.fn_affine_input(id))

    # Compute the varifold loss between all pairs of atlases with similarity transform,
    # running a quick registration between all pairs. This might get too much for large
    # training sets though
    ids = list(md_ds.keys())
    dsq_sub = np.zeros((len(md_ds), len(md_ds)))

    # The default affine parameters
    theta_all = { }
    kernel = GaussLinKernelWithLabels(sigma_varifold, md_ds[ids[0]].lp.shape[1])
    for i1, (k1,v1) in enumerate(md_ds.items()):
        for i2, (k2,v2) in enumerate(md_ds.items()):
            if k1 != k2: 
                # Define the symmetric loss for this pair
                loss_ab = lossVarifoldSurfWithLabels(v1.ft, v2.vt, v2.ft, v1.lpt, v2.lpt, kernel)
                loss_ba = lossVarifoldSurfWithLabels(v2.ft, v1.vt, v1.ft, v2.lpt, v1.lpt, kernel)
                pair_theta = torch.tensor([0.01, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0], 
                                        dtype=torch.float32, device=device, requires_grad=True)
                
                # Create optimizer
                opt_affine = torch.optim.LBFGS([pair_theta], max_eval=10, max_iter=10, line_search_fn='strong_wolfe')

                # Define closure
                def closure():
                    opt_affine.zero_grad()

                    R = rotation_from_vector(pair_theta[0:3]) * pair_theta[3]
                    b = pair_theta[4:]
                    R_inv = torch.inverse(R)
                    b_inv = - R_inv @ b

                    v1_to_v2 = (R @ v1.vt.t()).t() + b
                    v2_to_v1 = (R_inv @ v2.vt.t()).t() + b_inv

                    L = 0.5 * (loss_ab(v1_to_v2) + loss_ba(v2_to_v1))
                    L.backward()
                    return L
                
                # Run the optimization
                for i in range(10):
                    opt_affine.step(closure)

                # Print loss and record the best run/best parameters
                dsq_sub[i1,i2] = closure().item()
                theta_all[(k1,k2)] = pair_theta.detach()
                print(f'Pair {k1}, {k2} loss : {dsq_sub[i1,i2]:8.6f}')

    # Find the index of the template candidate
    i_best = np.argmin(dsq_sub.sum(axis=1))
    k_best = ids[i_best]
    print(f'Best template candidate: {k_best}, mean distance {dsq_sub.mean(axis=1)[i_best]} vs {dsq_sub.mean()}')

    # Now we need to go through and apply the best transformation to each case
    for i, (k,md_k) in enumerate(md_full.items()):
        affine_mat = np.eye(4)
        if k != k_best:
            # Compute the transform to move k to k_best
            theta_pair = theta_all[(k,k_best)]
            R = rotation_from_vector(theta_pair[0:3]) * theta_pair[3]
            b = theta_pair[4:]
            affine_mat[0:3,0:3] = R.detach().cpu().numpy()
            affine_mat[0:3,  3] = b.detach().cpu().numpy()

            # Save the registered mesh
            v_reg = (R @ md_k.vt.t()).t() + b
            pd_reg = vtk_make_pd(v_reg.detach().cpu().numpy(), md_k.f)
            pd_reg = vtk_set_cell_array(pd_reg, 'plab', md_k.lp)
            print(f'Saving {tbs.fn_affine_mesh(k)}')
            save_vtk(pd_reg, tbs.fn_affine_mesh(k))

            # Also saved the registered reduced mesh
            v_reg_ds = (R @ md_ds[k].vt.t()).t() + b
            pd_reg_ds = vtk_make_pd(v_reg_ds.detach().cpu().numpy(), md_ds[k].f)
            pd_reg_ds = vtk_set_cell_array(pd_reg_ds, 'plab', md_ds[k].lp)
            print(f'Saving {tbs.fn_affine_mesh_reduced(k)}')
            save_vtk(pd_reg_ds, tbs.fn_affine_mesh_reduced(k))

        else:
            pd_reg = vtk_make_pd(md_k.vt.detach().cpu().numpy(), md_k.f)
            pd_reg = vtk_set_cell_array(pd_reg, 'plab', md_k.lp)
            print(f'Saving {tbs.fn_affine_mesh(k)}')
            save_vtk(pd_reg, tbs.fn_affine_mesh(k))

            pd_reg_ds = vtk_make_pd(md_ds[k].vt.detach().cpu().numpy(), md_ds[k].f)
            pd_reg_ds = vtk_set_cell_array(pd_reg_ds, 'plab', md_ds[k].lp)
            print(f'Saving {tbs.fn_affine_mesh_reduced(k)}')
            save_vtk(pd_reg_ds, tbs.fn_affine_mesh_reduced(k))

        # Now save the affine matrix
        np.savetxt(tbs.fn_affine_matrix(k), affine_mat)


def shoot_root_to_template(md_root, p_root, sigma_root=20):
    device = md_root.vt.device
    K_root = GaussKernel(sigma=torch.tensor(sigma_root, dtype=torch.float32, device=device))
    _, q_temp = Shooting(p_root, md_root.vt.clone().requires_grad_(True).contiguous(), K_root)[-1]
    pd = vtk_make_pd(q_temp.detach().cpu().numpy(), md_root.f)
    pd = vtk_set_cell_array(pd, 'plab', md_root.lp)
    return MeshData(pd, device=q_temp.device)


def shoot_template_to_subject(md_temp, p, sigma_root=20):
    device = md_temp.vt.device
    K_root = GaussKernel(sigma=torch.tensor(sigma_root, dtype=torch.float32, device=device))
    _, q = Shooting(p, md_temp.vt.clone().requires_grad_(True).contiguous(), K_root)[-1]
    pd = vtk_make_pd(q.detach().cpu().numpy(), md_temp.f)
    pd = vtk_set_cell_array(pd, 'plab', md_temp.lp)
    return MeshData(pd, device=device)


# Update the template by remeshing and updating probability labels
def update_model_by_remeshing(md_root, md_targets, p_root, p_temp_z, 
                              sigma_lddmm=5, sigma_root=20, 
                              targetlen=1.0, featuredeg=30):

    # LDDMM kernels
    device = md_root.vt.device
    K_root = GaussKernel(sigma=torch.tensor(sigma_root, dtype=torch.float32, device=device))
    K_temp = GaussKernel(sigma=torch.tensor(sigma_lddmm, dtype=torch.float32, device=device))

    # Shoot from root to obtain the template
    q_root = md_root.vt.clone().requires_grad_(True).contiguous()
    _, q_temp = Shooting(p_root, q_root, K_root)[-1]
    pd_template = vtk_make_pd(q_temp.detach().cpu().numpy(), md_root.f)

    # Sample and average the plab array from the subjects using OMT
    plab_sample = []
    for i, (id, md_i) in enumerate(md_targets.items()):
        _, q_i = Shooting(p_temp_z[i,:], q_temp, K_temp)[-1]
        _, w_omt = match_omt(q_i, md_root.ft, md_i.vt, md_i.ft)
        lp_omt = vtk_sample_cell_array_at_vertices(md_i.pd, md_i.lp, w_omt.detach().cpu().numpy())
        plab_sample.append(lp_omt)
    plab_sample_avg = np.stack(plab_sample).mean(0)

    # Apply remeshing to the template
    v_remesh, f_remesh = isotropic_explicit_remeshing(
        q_temp.detach().cpu().numpy(), md_root.f, 
        targetlen=targetlen, featuredeg=featuredeg)
    pd_remesh = vtk_make_pd(v_remesh, f_remesh)

    # Apply the remeshing to the plab array
    _, w_omt = match_omt(torch.tensor(v_remesh, dtype=torch.float32, device=device), 
                         torch.tensor(f_remesh, dtype=torch.long, device=device),
                         q_temp, md_root.ft)
    lp_remesh = vtk_sample_cell_array_at_vertices(pd_template, plab_sample_avg, w_omt.detach().cpu().numpy())
    pd_remesh = vtk_set_cell_array(pd_remesh, 'plab', softmax(lp_remesh * 10, 1))

    # Return the new template as MeshData
    return MeshData(pd_remesh, device=md_root.vt.device)


# Map template into subject space and return mesh
def map_template_to_subject(md_temp, md_target, p_temp, sigma_lddmm=5):

    # LDDMM kernels
    device = md_temp.vt.device
    K_temp = GaussKernel(sigma=torch.tensor(sigma_lddmm, dtype=torch.float32, device=device))

    # Shoot from template to subject and save as polydata/meshdata
    q_temp = md_temp.vt.clone().requires_grad_(True).contiguous()
    _, q_fitted = Shooting(p_temp, q_temp, K_temp)[-1]
    pd_fitted = vtk_make_pd(q_fitted.detach().cpu().numpy(), md_temp.f)
    pd_fitted = vtk_set_cell_array(pd_fitted, 'plab', md_temp.lp)

    # Match the subject via OMT, i.e. every template vertex is mapped to somewhere on the
    # subject mesh, this fits more closely than LDDMM but might break topology
    # _, w_omt = match_omt(md_target.vt, md_target.ft, q_fitted, md_temp.ft)
    _, w_omt = match_omt(q_fitted, md_temp.ft, md_target.vt, md_target.ft)
    v_omt, v_int, w_int = omt_match_to_vertex_weights(pd_fitted, md_target.pd, w_omt.detach().cpu().numpy())

    # Create a clean model to return
    pd_omt = vtk_make_pd(v_omt, md_temp.f)
    pd_omt = vtk_set_cell_array(pd_omt, 'plab', md_temp.lp)
    pd_omt = vtk_set_cell_array(pd_omt, 'match', w_omt.detach().cpu().numpy())

    # Get the interpolation arrays from the matching and place them into the fitted template
    pd_fitted = vtk_set_point_array(pd_fitted, 'omt_v_int', v_int)
    pd_fitted = vtk_set_point_array(pd_fitted, 'omt_w_int', w_int)

    return pd_fitted, pd_omt


def template_from_ellipsoid_keops(tbs: TemplateBuildWorkspace, template: Template, device):

    # Read the downsampled affine-aligned meshes
    md_aff_ds = { id: MeshData(load_vtk(tbs.fn_affine_mesh_reduced(id)), device) for id,_ in tbs.get_subjects() }

    # Read the affine-aligned meshes
    md_aff = { id: MeshData(load_vtk(tbs.fn_affine_mesh(id)), device) for id,_ in tbs.get_subjects() }

    # Generate a sphere
    ms = pymeshlab.MeshSet()
    ms.create_sphere(subdiv = 4)
    m0 = ms.mesh(0)
    v_sph, f_sph = m0.vertex_matrix(), m0.face_matrix()
    pd_sph = vtk_make_pd(v_sph, f_sph)
    pd_sph = vtk_set_cell_array(pd_sph, 'plab', np.zeros((f_sph.shape[0],1)))
    md_sph = MeshData(pd_sph, device)

    # Map the cartesian coordinates to spherical coordinates
    sph_phi = np.arctan2(v_sph[:,1], v_sph[:,0])
    sph_theta = np.arccos(v_sph[:,2])

    # Find an affine transformation of the sphere that best aligns with the data 
    # using the varifold measure
    v_all = np.concatenate([ x.v for id,x in md_aff_ds.items() ], 0)
    pca = PCA(n_components=3)
    pca.fit(v_all)

    # Create losses for each of the target meshes
    kernel = GaussLinKernel(torch.tensor([template.get_varifold_sigma()], dtype=torch.float32, device=device))
    loss = { id: lossVarifoldSurf(md_sph.ft, md.vt, md.ft, kernel) for (id,md) in md_aff_ds.items() }

    # Create a parameter tensor for the sphere
    b = torch.tensor(pca.mean_, dtype=torch.float32, device=device, requires_grad=True).contiguous()
    A = torch.tensor(pca.get_covariance(), dtype=torch.float32, device=device, requires_grad=True).contiguous()

    # Generate a combined objective function
    optimizer = torch.optim.LBFGS([A,b], max_eval=16, max_iter=16, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # Apply transformation to the sphere
        x = md_sph.vt
        y = (A @ x.T).T + b
        L = 0
        for i, (id,v) in enumerate(md_aff_ds.items()):
            L = L + loss[id](y)
        L = L / len(md_aff_ds.items())
        L.backward()
        return L

    print('Affine fitting ellipsoid to population')
    for i in range(30):
        print(f'Iter {i:03d}, Loss: {closure()}')
        optimizer.step(closure)

    # Compute the new rotated sphere
    v_sph_opt = (A @ md_sph.vt.T).T + b

    # Peform remeshing of the sphere
    targetlen = template.get_remeshing_edge_len_pct()
    featuredeg = template.get_remeshing_feature_angle()
    v_ell, f_ell = isotropic_explicit_remeshing(
        v_sph_opt.detach().cpu().numpy(), md_sph.f, 
        targetlen=PyMeshLabInterface.percentage(targetlen), featuredeg=featuredeg)

    pd_ell = vtk_make_pd(v_ell, f_ell)
    pd_ell = vtk_set_cell_array(pd_ell, 'plab', np.zeros((f_ell.shape[0],1)))
    md_ell = MeshData(pd_ell, device)

    # Sample the label probabilities from the target shapes (full resolution) using OMT
    plab_sample = []
    for _, md_i in md_aff.items():
        _, w_omt = match_omt(md_ell.vt, md_ell.ft, md_i.vt, md_i.ft)
        lp_omt = vtk_sample_cell_array_at_vertices(md_i.pd, md_i.lp, w_omt.detach().cpu().numpy())
        plab_sample.append(lp_omt)
    plab_sample_avg = np.stack(plab_sample).mean(0)

    # Save the best fit ellipsoid
    pd_sphere_opt = vtk_make_pd(v_ell, f_ell)
    vtk_set_cell_array(pd_sphere_opt, 'plab', softmax(plab_sample_avg * 10, axis=1))
    save_vtk(pd_sphere_opt, f'{tbs.lddmm_dir()}/ellipsoid_best_fit.vtk')

    # Use the ellipsoid as the new template
    md_root = MeshData(pd_sphere_opt, device)

    # Get the parameters from the template json
    sigma_lddmm = template.get_lddmm_sigma()
    sigma_root = template.get_build_root_sigma_factor() * sigma_lddmm
    sigma_varifold = template.get_varifold_sigma()
    w_jacobian_penalty = template.get_jacobian_penalty()
    gamma_lddmm = template.get_lddmm_gamma()
    nt = template.get_lddmm_nt()

    # Iterate over the schedule
    for i, iter in enumerate(template.get_build_iteration_schedule()):

        # Print iteration
        print(f'*** TEMPLATE BUILD STAGE {i} ***')

        # Fit the model to the population (use downsampled target meshes)  
        p_root, p_temp = fit_model_to_population(
            md_root, md_aff_ds, iter, nt,
            sigma_lddmm=sigma_lddmm, sigma_root=sigma_root, 
            sigma_varifold=sigma_varifold, gamma_lddmm=gamma_lddmm,
            w_jacobian_penalty=w_jacobian_penalty) 

        # Compute the template by forward shooting
        md_temp = shoot_root_to_template(md_root, p_root, sigma_root=sigma_root)
        save_vtk(md_temp.pd, f'{tbs.lddmm_dir()}/template_iter{i:02d}.vtk')

        # Remesh the template (use fill-resolution target meshes to sample plab)
        md_remesh = update_model_by_remeshing(
            md_root, md_aff, p_root, p_temp, 
            sigma_lddmm=sigma_lddmm, sigma_root=sigma_root, 
            targetlen=PyMeshLabInterface.percentage(targetlen), featuredeg=featuredeg)
        save_vtk(md_remesh.pd, f'{tbs.lddmm_dir()}/template_iter{i:02d}_remesh.vtk')

        # Make the template the new root
        md_root = md_remesh

    # Save the template and the momenta
    pd_temp_save = vtk_clone_pd(md_temp.pd)
    for i, (id, md_i) in enumerate(md_aff.items()):
        vtk_set_point_array(pd_temp_save, f'momenta_{id}', p_temp[i,:,:].detach().cpu().numpy())
    save_vtk(pd_temp_save, f'{tbs.lddmm_dir()}/template_final_with_momenta.vtk')


def generate_template_output_folder(tbs: TemplateBuildWorkspace, template: Template, out_dir):

    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # Read the template with the momenta
    pd_temp = load_vtk(f'{tbs.lddmm_dir()}/template_final_with_momenta.vtk')

    # We only want to keep the plab array, not the momenta that have input ids
    v,f,lp = vtk_get_points(pd_temp), vtk_get_triangles(pd_temp), vtk_get_cell_array(pd_temp, 'plab')
    pd_left = vtk_make_pd(v, f)
    vtk_set_cell_array(pd_left, 'plab', lp)

    # Compute curvature measures on the template
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))

    # Compute curvatures (not that important)
    for c_id,c_nm in { 0: 'Mean', 1: 'Gaussian', 4: 'ShapeIndex', 5: 'Curvedness' }.items():
        ms.compute_curvature_principal_directions_per_vertex(
            method='Scale Dependent Quadric Fitting', 
            curvcolormethod=c_id,
            scale=pymeshlab.AbsoluteValue(3.0))
        q = ms.mesh(0)
        vtk_set_point_array(pd_left, f'Curvature_{c_nm}', ms.mesh(0).vertex_scalar_array())

    # Set the label array in the template by taking argmax over plab
    vtk_set_cell_array(pd_left, 'label', np.argmax(lp, axis=1))

    # Save the left template
    save_vtk(pd_left, f'{out_dir}/template_shoot_left.vtk')

    # Apply a flip to the left template
    flip_lr = np.loadtxt(os.path.join(template.root, 'ashs_template_flip.mat'))
    pd_right = vtk_clone_pd(pd_left)
    vtk_set_points(pd_right, np.einsum('ij,kj->ki', flip_lr[:3,:3].T, v) + flip_lr[:3,3])
    save_vtk(pd_right, f'{out_dir}/template_shoot_right.vtk')

    # Save the JSON
    with open(f'{out_dir}/template.json','wt') as fd:
        json.dump(template.json, fd)


def finalize_groupwise_keops(tbs: TemplateBuildWorkspace, template: Template, device):

    # Read the template with the momenta
    pd_temp = load_vtk(f'{tbs.lddmm_dir()}/template_final_with_momenta.vtk')
    md_temp = MeshData(pd_temp, device)

    # Read the affine-aligned meshes
    md_aff = { id: MeshData(load_vtk(tbs.fn_affine_mesh(id)), device) for id,_ in tbs.get_subjects() }

    # Set the parameters
    sigma_lddmm = template.get_lddmm_sigma()
    nt = template.get_lddmm_nt()

    # For each mesh, save the fitted template and the OMT matching to the average surface
    K_temp = GaussKernel(sigma=torch.tensor(sigma_lddmm, dtype=torch.float32, device=device))
    for i, (id, md_i) in enumerate(md_aff.items()):
        
        # Save the subject's momenta in template space
        ws_i = tbs.get_subject_workspace(id)
        p_i_np = vtk_get_point_array(pd_temp, f'momenta_{id}')
        pd = vtk_make_pd(md_temp.v, md_temp.f)
        vtk_set_point_array(pd, 'Momentum', p_i_np)
        vtk_set_field_data(pd, 'lddmm_sigma', sigma_lddmm)
        vtk_set_field_data(pd, 'lddmm_nt', nt)        
        save_vtk(pd, ws_i.fit_lddmm_momenta)

        # Create the fitted model in the space of the inflated model, in RAS space
        q_temp = md_temp.vt.clone().requires_grad_(True).contiguous()
        p_i = torch.tensor(p_i_np, dtype=torch.float32, device=device).requires_grad_(True).contiguous()
        _, q_fitted = Shooting(p_i, q_temp, K_temp, nt)[-1]
        pd_fitted = vtk_make_pd(q_fitted.detach().cpu().numpy(), md_temp.f)
        vtk_set_cell_array(pd_fitted, 'plab', md_temp.lp)

        # Perform OMT matching to the target shape and then to native space
        pd_subj_native = load_vtk(ws_i.fn_cruise(f'mtl_avg_l2m-mesh.vtk'))
        pd_omt_to_infl, pd_omt_to_native = omt_match_fitted_template_to_target(pd_fitted, md_i.pd, pd_subj_native, device)
        
        # Save the fitted, omt, and omt-to-native meshes
        save_vtk(pd_fitted, tbs.fn_final_fitted_mesh(id))
        save_vtk(pd_omt_to_infl, tbs.fn_final_omt_mesh(id))

        # Propagate this fitted mesh through the levelset layers using OMT
        print(f'Propagating through level set {id}')
        img_ls = sitk.ReadImage(ws_i.fn_cruise('mtl_layering-boundaries.nii.gz'))
        prof_meshes, mid_layer = profile_meshing_omt(img_ls, source_mesh=pd_omt_to_native, device=device)

        # Get the sform for this subject
        sform = np.loadtxt(ws_i.cruise_sform)
 
        # Save these profile meshes, but mapping to RAS coordinate space for compatibility
        # with image sampling tools
        vtk_apply_sform(pd_omt_to_native, sform)
        save_vtk(pd_omt_to_native, tbs.fn_final_omt_to_native_hw_mesh(id))
        for layer, pd_layer in enumerate(prof_meshes):
            vtk_apply_sform(pd_layer, sform, cell_coord_arrays=['wmatch'])
            save_vtk(pd_layer, ws_i.fn_fit_profile_mesh(layer), binary=True)




"""
OLD CODE
def groupwise_lddmm(tbs: TemplateBuildWorkspace, template: Template, device, stages=5, eta=0.5):

    # Read the affine meshes
    ids = tbs.get_ids()
    md_aff = {}
    for id in ids:
        md_aff[id] = MeshData(load_vtk(tbs.fn_affine_mesh(id)), device)

    # Sigma parameters for losses
    sigma_varifold = torch.tensor([template.get_varifold_sigma()], dtype=torch.float32, device=device)
    sigma_lddmm = torch.tensor([template.get_lddmm_sigma()], dtype=torch.float32, device=device)

    # Compute the varifold loss between all pairs of atlases in the template subset before
    # any registration - this is to determine the best candidate for the template
    dsq_sub_aff = np.zeros((len(md_aff), len(md_aff)))
    kernel = GaussLinKernelWithLabels(sigma_varifold, md_aff[ids[0]].lp.shape[1])
    for i1, (k1,v1) in enumerate(md_aff.items()):
        for i2, (k2,v2) in enumerate(md_aff.items()):
            pair_loss = lossVarifoldSurfWithLabels(v1.ft, v2.vt, v2.ft, v1.lpt, v2.lpt, kernel)
            dsq_sub_aff[i1,i2] = pair_loss(v1.vt).item()
            
    # Find the index of the template candidate
    i_src = np.argmin(dsq_sub_aff.sum(axis=1))
    id_src = ids[i_src]
    print(f'Best template candidate: {id_src}, mean distance {dsq_sub_aff.mean(axis=1)[i_src]} vs {dsq_sub_aff.mean()}')

    # Select this candidate and go with it
    md_src = md_aff[id_src]

    # Create losses for each of the target meshes
    loss = {}
    for i, (id,v) in enumerate(md_aff.items()):
        dataloss = lossVarifoldSurfWithLabels(md_src.ft, v.vt, v.ft, md_src.lpt, v.lpt, kernel)
        loss[id] = LDDMMloss(GaussKernel(sigma=sigma_lddmm), dataloss)

    # Save the initial template
    save_vtk(md_src.pd, tbs.fn_template_build_mesh(0))

    # Create a storage for the template coordinates and momenta at each iteration
    td = [{
        'q': torch.tensor(md_src.vt, dtype=torch.float32, device=device, requires_grad=True).contiguous(),
        'p': torch.zeros((len(md_aff),) + md_src.vt.shape, dtype=torch.float32, device=device, requires_grad=True).contiguous()
        }]

    # Outer loop: template update iterations
    for m in range(stages):

        # Generate a combined objective function
        optimizer = torch.optim.LBFGS([td[m]['p']], max_eval=16, max_iter=16, line_search_fn='strong_wolfe')
        print(f'*** STAGE {m} OPTIMIZATION ***')
        start = time.time()

        def closure():
            optimizer.zero_grad()
            L = 0
            for i, (id,v) in enumerate(md_aff.items()):
                L = L + loss[id](td[m]['p'][i,:], td[m]['q'])
            L = L / len(md_aff.items())
            L.backward()
            print(L.item())
            return L

        loss_history = []
        for i in range(10):
            loss_value = closure().item()
            loss_history.append(loss_value)
            print(f'Stage {m:02d}  Iter {i:03d}  loss : {loss_value}')
            optimizer.step(closure)

        print(f'Stage {m} (L-BFGS) time: {round(time.time() - start, 2)} seconds')
        td[m]['loss_history'] = np.array(loss_history)

        # Warp the template to each subject and save
        k_gauss = GaussKernel(sigma=sigma_lddmm)
        for i, (id,v) in enumerate(md_aff.items()):
            _, q_warp = Shooting(td[m]['p'][i], td[m]['q'], k_gauss)[-1]
            pd_warp = vtk_make_pd(q_warp.detach().cpu().numpy(), md_src.f)
            pd_warp = vtk_set_cell_array(pd_warp, 'plab', md_src.lp)
            save_vtk(pd_warp, tbs.fn_template_build_tosubj(m, id))

        # Compute the new starting point for optimization
        _, q_mean = Shooting(td[m]['p'].mean(axis=0) * eta, td[m]['q'], k_gauss)[-1]

        # Add this new starting point to the accumulation list
        td.append({
            'q': q_mean.detach().clone().detach().requires_grad_(True),
            'p': torch.zeros_like(td[m]['p']).requires_grad_(True)
        })

        # Save the template with labels
        v_opt = q_mean.detach().cpu().numpy()
        pd_template = vtk_make_pd(v_opt, md_src.f)
        pd_template = vtk_set_cell_array(pd_template, 'plab', md_src.lp)

        # Save the indivifual momenta in the template as well for future use
        for i, (id,v) in enumerate(md_aff.items()):
            p_i = td[m]['p'][i].detach().cpu().numpy()
            pd_template = vtk_set_point_array(pd_template, f'p_{id}', p_i)

        # Save the left side template
        save_vtk(pd_template, tbs.fn_template_build_mesh(m+1))

        # Generate a plot of the loss
        for i, tdi in enumerate(td[:-1]):
            plt.plot(tdi['loss_history'], label=f'Stage {i}')
        plt.title('Loss function by stage')
        plt.xlabel('Iteration')
        plt.ylabel('Loss value')
        plt.savefig(os.path.join(tbs.qa_dir(), 'loss_by_stage.png'))
"""


class BuildTemplateLauncher:

    def __init__(self, parse):

        # Add the arguments
        parse.add_argument('template_init_dir', help='Template initial directory structure', type=pathlib.Path)
        parse.add_argument('ashs_json', help='JSON file desribing the input files', type=argparse.FileType('rt'))
        parse.add_argument('work_dir', metavar='work_dir', type=str, help='Working directory')
        parse.add_argument('output_dir', metavar='output_dir', type=str, help='Template output directory')
        parse.add_argument('-f', '--fusion-stage', type=str, choices=['multiatlas', 'bootstrap'], 
                        help='Which stage of ASHS fusion to select', default='bootstrap')                   
        parse.add_argument('-c', '--correction-mode', type=str, choices=['heur', 'corr_usegray', 'corr_nogray'], 
                        help='Which ASHS correction output to select', default='corr_usegray')                   
        parse.add_argument('-d', '--device', type=str, 
                        help='PyTorch device to use (cpu, cuda0, etc)', default='cpu')
        parse.add_argument('--skip-cruise', action='store_true')
        parse.add_argument('--skip-affine', action='store_true')

        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))

    
    def run(self, args):

        # Load the template
        template = Template(args.template_init_dir)

        # Load the ASHS json
        ashs_input_desc = json.load(args.ashs_json)

        # Prepare device
        # device = torch.device(args.device) if torch.cuda.is_available() else 'cpu'
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        print("Is cuda available?", torch.cuda.is_available())
        print("Device count?", torch.cuda.device_count())
        print("Current device?", torch.cuda.current_device())
        print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))

        # Keep track of ASHS importers and workspaces created
        tbs = TemplateBuildWorkspace(args.work_dir)

        # Run basic Nighres for each subject
        for d in ashs_input_desc:
            id = d['id']
            side = d['side']

            # There are two ways to do this. One is that the user provides the ASHS directory
            # in 'ashs_dir' (or for back-compatibility, 'path') and then we run CRUISE on this
            # directory and output it into the work directory. Another is that the user supplies
            # and existing workspace for each subject where CRUISE has already been run
            if 'crashs_dir' in d:
                out_dir = d['crashs_dir']
                workspace = Workspace(out_dir, id, side)
            else:
                # Create the output dir
                out_dir = os.path.join(args.work_dir, id)
                os.makedirs(out_dir, exist_ok=True)
                workspace = Workspace(out_dir, id, side)

                # Run the CRUISE part
                if not args.skip_cruise:

                    # Load the ASHS experiment
                    ashs_dir = d['ashs_dir'] if 'ashs_dir' in d else d['path']
                    ashs = ASHSFolder(ashs_dir, side, args.fusion_stage, args.correction_mode)
                    ashs.load_posteriors(template)

                    # Convert the inputs into probability maps
                    print("Converting ASHS-T1 posteriors to CRUISE inputs")
                    ashs_output_to_cruise_input(template, ashs, workspace)

                    # Run CRUISE on the inputs
                    print("Running CRUISE to correct topology and compute inflated surface")
                    run_cruise(workspace, template, overwrite=True)

                    # Perform postprocessing
                    print("Mapping ASHS labels on the inflated template")
                    cruise_postproc(template, ashs, workspace, 
                                    reduction=template.get_cruise_inflate_reduction())

            # Store the data
            tbs.add_subject(id, workspace, side)

        # The LDDMM portion
        if not args.skip_affine:
            groupwise_similarity_registration_keops(tbs, template, device=device)

        # Run the groupwise code
        template_from_ellipsoid_keops(tbs, template, device)
        generate_template_output_folder(tbs, template, args.output_dir  )
        finalize_groupwise_keops(tbs, template, device=device)

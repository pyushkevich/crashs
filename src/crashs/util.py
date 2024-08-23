import torch
import scipy
import SimpleITK as sitk
from picsl_c3d import Convert3D
import os
import glob
import json

from crashs.vtkutil import *

# Class that represents a surface mesh for use in PyTorch
class MeshData:
    def __init__(self, pd:vtk.vtkPolyData, device, target_faces=None, transform=None):
        self.pd = vtk_all_point_arrays_to_cell_arrays(pd)
        self.v, self.f = vtk_get_points(self.pd), vtk_get_triangles(self.pd)
        self.lp = vtk_get_cell_array(self.pd, 'plab')
        if target_faces is not None:
            # Perform decimation
            vd, fd = decimate(self.v, self.f, target_faces)

            # Get new cell probability labels
            pd_tmp = vtk_make_pd(vd, fd)
            vtk_set_point_array(pd_tmp, 'plab', vtk_sample_cell_array_at_vertices(self.pd, self.lp, vd))
            pd_tmp = vtk_all_point_arrays_to_cell_arrays(pd_tmp)

            self.v, self.f = vd, fd
            self.lp = vtk_get_cell_array(pd_tmp, 'plab')
        if transform is not None:
            self.v = np.einsum('ij,kj->ki', transform[:3,:3], self.v) + transform[:3,3]

        self.device = device
        self.vt = torch.tensor(self.v, dtype=torch.float32, device=device).contiguous()
        self.ft = torch.tensor(self.f, dtype=torch.long, device=device).contiguous()
        self.lpt = torch.tensor(self.lp, dtype=torch.float32, device=device).contiguous()

    def apply_transform(self, transform):
        vtk_apply_sform(self.pd, transform)
        self.v = vtk_get_points(self.pd)
        self.vt = torch.tensor(self.v, dtype=torch.float32, device=self.device).contiguous()

    def export(self, fn):
        pd_reduced = vtk_make_pd(self.v, self.f)
        vtk_set_cell_array(pd_reduced, 'plab', self.lp)
        vtk_set_cell_array(pd_reduced, 'label', np.argmax(self.lp, axis=1))
        save_vtk(pd_reduced, fn)


# Class that represents information about a template
class Template :
    def __init__(self, template_dir):
        # Store the template directory
        self.root = template_dir

        # Load the template json
        with open(os.path.join(template_dir, 'template.json')) as template_json:
            self.json = json.load(template_json)

    def get_mesh(self, side):
        return os.path.join(self.root, self.json['sides'][side]['mesh'])

    def get_labels_for_tissue_class(self, tissue_class):
        return self.json['ashs_label_type'][tissue_class]
    
    def get_label_types(self):
        return self.json['ashs_label_type']
    
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
    
    def get_jacobian_penalty(self):
        return self.json['registration'].get('jacobian_penalty', 0.)

    def get_cruise_inflate_maxiter(self):
        return self.json.get('cruise', dict()).get('inflate_maxiter', 500)

    def get_cruise_laminar_n_layers(self):
        return self.json.get('cruise', dict()).get('laminar_n_layers', 4)

    def get_cruise_laminar_method(self):
        return self.json.get('cruise', dict()).get('laminar_method', 'distance-preserving')
    
    def get_cruise_inflate_reduction(self):
        return self.json.get('cruise', dict()).get('inflate_reduction', None)

    def get_remeshing_edge_len_pct(self):
        return self.json.get('template_build', dict()).get('remeshing_edge_length_pct', 1.0)
    
    def get_build_iteration_schedule(self):
        return self.json.get('template_build', dict()).get('schedule', [10,10,10,50])

    def get_build_root_sigma_factor(self):
        return self.json.get('template_build', dict()).get('root_sigma_factor', 2.4)
    
    def get_preprocessing_mode(self):
        return self.json.get('preprocessing', dict()).get('mode', None)


def find_unique_file_with_suffix(dir, suffix):
    if os.path.isdir(dir):
        matches = [ f for f in os.listdir(dir) if f.endswith(suffix) ]
        if len(matches) == 1:
            return os.path.join(dir, matches[0])
    return None


def find_file(fullpath):
    return fullpath if os.path.exists(fullpath) else None

        
# Class to represent an ASHS output folder and the files that we need from
# ASHS for this script
class ASHSFolder:

    def __init__(self, ashs_dir, side, fusion_mode, correction_mode):

        self.ashs_dir = ashs_dir
        self.side = side
        self.fusion_mode = fusion_mode
        self.correction_mode = correction_mode

        # How posteriors are coded
        pstr = 'posterior' if correction_mode == 'heur' else f'posterior_{correction_mode}'

        # Set the posterior pattern
        self.posterior_pattern = os.path.join(
            ashs_dir, 
            f'{fusion_mode}/fusion/{pstr}_{side}_%03d.nii.gz')
        
        # Find the MPRAGE
        self.mprage = find_file(f'{ashs_dir}/mprage.nii.gz')
        
        # Find the final segmentation if the posteriors are not available
        self.final_seg = find_unique_file_with_suffix(f'{ashs_dir}/final', f'_{side}_lfseg_{correction_mode}.nii.gz')

        # Find the TSE native chunk image
        self.tse_native_chunk = find_unique_file_with_suffix(f'{ashs_dir}', f'tse_native_chunk_{side}.nii.gz')

        # Required matrix files
        self.affine_to_template = find_file(f'{ashs_dir}/affine_t1_to_template/t1_to_template_affine.mat')
        self.affine_t2f_t1m = find_file(f'{ashs_dir}/flirt_t2_to_t1/flirt_t2_to_t1.mat')
        self.affine_t1f_t2m = find_file(f'{ashs_dir}/flirt_t2_to_t1/flirt_t2_to_t1_inv.mat')

    def set_alternate_posteriors(self, pattern):
        self.posterior_pattern = pattern
        
    def load_posteriors(self, template:Template):

        # Load the posteriors from posterior files
        self.posteriors = {}
        for lab in ['wm', 'gm', 'bg']:
            for v in template.get_labels_for_tissue_class(lab):
                img=self.posterior_pattern % (v,)
                if os.path.exists(img):
                    self.posteriors[v] = sitk.ReadImage(img)

        # If the posteriors do not exist, load them from final segmentation instead
        if len(self.posteriors) == 0:
            c3d = Convert3D()
            c3d.execute(f'{self.tse_native_chunk} {self.final_seg} -int 0 -reslice-identity -popas X')
            for lab in ['wm', 'gm', 'bg']:
                for v in template.get_labels_for_tissue_class(lab):
                    c3d.execute(f'-push X -thresh {v} {v} 1 0')
                    self.posteriors[v] = c3d.pop()

    
class Workspace:

    # Define output files
    def fn_cruise(self, suffix):
        return os.path.join(self.cruise_dir, f'{self.expid}_{suffix}')

    def fn_fit(self, suffix):
        return os.path.join(self.fit_dir, f'{self.expid}_{suffix}')

    def fn_thick(self, suffix):
        return os.path.join(self.thick_dir, f'{self.expid}_{suffix}')

    def __init__(self, output_dir, expid, side):
        self.output_dir = output_dir
        self.expid = expid
        self.side = side

        # Define and create output directories
        self.cruise_dir = os.path.join(self.output_dir, 'cruise')
        self.fit_dir = os.path.join(self.output_dir, 'fitting')
        self.thick_dir = os.path.join(self.output_dir, 'thickness')
        os.makedirs(self.cruise_dir, exist_ok=True)
        os.makedirs(self.fit_dir, exist_ok=True)
        os.makedirs(self.thick_dir, exist_ok=True)

        self.cruise_fn_base = f'{self.expid}_mtl'
        self.cruise_wm_prob = self.fn_cruise('wm_prob.nii.gz')
        self.cruise_gm_prob = self.fn_cruise('gm_prob.nii.gz')
        self.cruise_bg_prob = self.fn_cruise('bg_prob.nii.gz')
        self.cruise_wm_mask = self.fn_cruise('wm_mask_lcomp.nii.gz')
        self.cruise_infl_mesh = self.fn_cruise('mtl_avg_infl-mesh.vtk')
        self.cruise_infl_mesh_labeled = self.fn_cruise('mtl_avg_infl-mesh-ras-labeled.vtk')
        self.cruise_middepth_mesh = self.fn_cruise('mtl_avg_l2m-mesh.vtk')
        self.cruise_sform = self.fn_cruise('sform.mat')

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

        self.thick_boundary = self.fn_thick('thick_boundary_raw.vtk')
        self.thick_boundary_sm = self.fn_thick('thick_boundary_smooth.vtk')
        self.thick_tetra_mesh = self.fn_thick('thickness_tetra.vtk')
        self.thick_skeleton = self.fn_thick('skeleton.vtk')
        self.thick_result = self.fn_thick('template_thickness.vtk')

    def fn_fit_profile_mesh(self, k:int):
        return self.fn_fit(f'fitted_omt_match_to_p{k:02d}.vtk')

    def fn_fit_profile_meshes(self):
        return glob.glob(self.fn_fit(f'fitted_omt_match_to_p??.vtk'))


# Load the available label posteriors into a dictionary
#def load_ashs_posteriors(template:Template, posterior_pattern):
#    p = {}
#    for lab in ['wm', 'gm', 'bg']:
#        for v in template.get_labels_for_tissue_class(lab):
#            img=posterior_pattern % (v,)
#            if os.path.exists(img):
#                p[v] = sitk.ReadImage(img)
#    return p

# Routine that takes ASHS posteriors and combines them into a tissue probability map
def ashs_posteriors_to_tissue_probabilities(
        ashs_posteriors: dict, category_labels: dict, 
        category_order: list, background_category, 
        softmax_scale=10):
    
    # Load the posteriors and compute maximum probability per voxel
    idat = {}
    for lab in category_order:
        idat[lab] = {}
        prob_max=None
        for v in category_labels[lab]:
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
    bix = category_order.index(background_category)
    x = np.concatenate(
        [idat[lab]['prob_max'][:, :, :, np.newaxis] for lab in category_order], 
        axis=3)

    # Replace missing data with background probability
    x[:,:,:,bix] = np.where(x.max(axis=3) == 0., 1.0, x[:,:,:,bix])

    # Compute the softmax
    y = scipy.special.softmax(x * softmax_scale, axis=3)

    # Return the results
    return idat, y


# Routine that takes ASHS posteriors and combines them into a tissue label image
def ashs_posteriors_to_tissue_labels(
        ashs_posteriors: dict, category_labels: dict, 
        category_order: list, background_category, 
        softmax_scale=10):
    idat, prob = ashs_posteriors_to_tissue_probabilities(
        ashs_posteriors, category_labels, category_order, background_category, softmax_scale)    
    tseg = sitk.GetImageFromArray(np.argmax(prob,3))
    tseg.CopyInformation(idat[background_category]['itk'])
    lab_to_idx = { lab:i for (i, lab) in enumerate(category_order) }
    return tseg, lab_to_idx
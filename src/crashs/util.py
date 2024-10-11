import torch
import scipy
import SimpleITK as sitk
from picsl_c3d import Convert3D
import os
import glob
import json
import pkg_resources
import copy
import shutil

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


# CRASHS main data folder - contains templates and trained models for fitting, preprocessing
class CrashsDataRoot : 

    def __init__(self, user_path=None):
        self.path = os.environ.get('CRASHS_DATA','') if user_path is None else user_path

    def find_template(self, template_id):
        for x in self.path.split(os.pathsep):
            json_path = os.path.join(x,'templates',template_id,'template.json')
            if os.path.exists(json_path):
                return os.path.join(x,'templates',template_id)
        raise FileNotFoundError(f'Template {template_id} not found, set search path using -C or CRASHS_DATA')
    
    def find_model(self, model_id):
        for x in self.path.split(os.pathsep):
            json_path = os.path.join(x, 'models', model_id, 'config.json')
            if os.path.exists(json_path):
                return os.path.join(x, 'models', model_id)
        raise FileNotFoundError(f'Model {model_id} not found, set search path using -C or CRASHS_DATA')


# Recursive dictionary merge
def merge_dicts(defaults, overrides):
    result = copy.deepcopy(defaults) 
    for k, v in overrides.items():
        if isinstance(v, dict):
            if k in result:
                if isinstance(result[k], dict):
                    result[k] = merge_dicts(result[k], v)
                else:
                    raise ValueError(f'Dictionary merge failure for key {k}')
            else:
                result[k] = v
        else:
            result[k] = v

    return result


# Class that represents information about a template
class Template :
    def __init__(self, template_dir):
        self.root = template_dir

        # Load the default parameters from the json in CRASHS source tree
        default_json_fn = pkg_resources.resource_filename('crashs', 'param/template_defaults.json')
        print('Merged template parameters: ', default_json_fn)
        with open(default_json_fn) as fn:
            self.json = json.load(fn)

        # Load the template-specific json
        local_json_fn = os.path.join(template_dir, 'template.json')
        if os.path.exists(local_json_fn):
            with open(local_json_fn) as template_json:
                self.json = merge_dicts(self.json, json.load(template_json))     
        else:
            print(f'Template directory does not contain template.json file. Using default settings.')


    def get_mesh(self, side):
        return os.path.join(self.root, self.json['sides'][side]['mesh'])
    
    def get_reduced_mesh_for_lddmm(self, side):
        mesh = self.json['sides'][side].get('mesh_reduced', None)
        return None if mesh is None else os.path.join(self.root, mesh)

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
    
    def get_lddmm_maxiter(self):
        return self.json['registration'].get('lddmm_maxiter', 150)
    
    def get_affine_maxiter(self):
        return self.json['registration'].get('affine_maxiter', 50)
    
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
    
    def get_remeshing_feature_angle(self):
        return self.json.get('template_build', dict()).get('remeshing_feature_angle', 30)
    
    def get_build_iteration_schedule(self):
        return self.json.get('template_build', dict()).get('schedule', [10,10,10,50])

    def get_build_root_sigma_factor(self):
        return self.json.get('template_build', dict()).get('root_sigma_factor', 2.4)
    
    def get_preprocessing_mode(self):
        return self.json.get('preprocessing', dict()).get('mode', None)

    def get_white_matter_nnunet_model(self):
        return self.json.get('preprocessing', dict()).get('nnunet_wm', None)


def find_unique_file_with_suffix(dir, suffix, missing='e'):
    if os.path.isdir(dir):
        matches = [ f for f in os.listdir(dir) if f.endswith(suffix) ]
        if len(matches) == 1:
            return os.path.join(dir, matches[0])
    if missing == 'q':
        return None
    elif missing == 'w':
        print(f'Warning: no file ending with {suffix} found in {dir}')
    else:
        raise FileNotFoundError(f'No file ending with {suffix} found in {dir}')
    

# Find file with optional error reporting
def find_file(fullpath, missing='e'):
    if os.path.exists(fullpath):
        return fullpath
    elif missing == 'q':
        return None
    elif missing == 'w':
        print(f'Warning: file {fullpath} not found')
    else:
        raise FileNotFoundError(f'File {fullpath} not found')

        
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

        # Locate the posteriors
        self.posterior_pattern = os.path.join(ashs_dir, f'{fusion_mode}/fusion/{pstr}_{side}_%03d.nii.gz')
                
        # Find the MPRAGE
        self.mprage = find_file(f'{ashs_dir}/mprage.nii.gz','w')
        
        # Find the final segmentation if the posteriors are not available
        self.final_seg = find_unique_file_with_suffix(f'{ashs_dir}/final', f'_{side}_lfseg_{correction_mode}.nii.gz','w')

        # Find the TSE native chunk image
        self.tse_native_chunk = find_file(f'{ashs_dir}/tse_native_chunk_{side}.nii.gz','w')

        # Required matrix files
        self.affine_to_template = find_file(f'{ashs_dir}/affine_t1_to_template/t1_to_template_affine.mat','e')
        self.affine_t2f_t1m = find_file(f'{ashs_dir}/flirt_t2_to_t1/flirt_t2_to_t1.mat','w')
        self.affine_t1f_t2m = find_file(f'{ashs_dir}/flirt_t2_to_t1/flirt_t2_to_t1_inv.mat','w')

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
            if self.final_seg is None:
                raise FileNotFoundError('No posteriors or final segmentation found in ASHS folder')
            
            # Typically we want to crop the final segmentation to the final chunk but if we don't have
            # that, then we use the image without cropping
            c3d = Convert3D()
            if self.tse_native_chunk is not None:
                c3d.execute(f'{self.tse_native_chunk} {self.final_seg} -int 0 -reslice-identity -popas X')
            else:
                c3d.execute(f'{self.final_seg} -trim 5mm -popas X')
            for lab in ['wm', 'gm', 'bg']:
                for v in template.get_labels_for_tissue_class(lab):
                    c3d.execute(f'-push X -thresh {v} {v} 1 0')
                    post = c3d.pop()
                    if np.count_nonzero(sitk.GetArrayFromImage(post)) > 0:
                        self.posteriors[v] = post
                        

# This method creates a dummy ASHS folder from a single segmentation
def make_ashs_folder(fn_segmentation:str, id:str, side:str, fn_output_dir:str,
                     fn_tse_chunk:str=None, fn_mprage:str=None,
                     fn_affine_to_template=None, 
                     correction_mode='heur'):
    
    
    # Copy the segmentation with the correct name into the folder
    c3d = Convert3D()
    for sub in 'final', 'flirt_t2_to_t1', 'affine_t1_to_template':
        os.makedirs(f'{fn_output_dir}/{sub}', exist_ok=True)
    
    # Save the segmentation
    fn_out_seg = f'{fn_output_dir}/final/{id}_{side}_lfseg_{correction_mode}.nii.gz'
    c3d.execute(f'{fn_segmentation} -type short -o {fn_out_seg}')
    
    # Save the TSE and MPRAGE
    if fn_tse_chunk:
        c3d.execute(f'{fn_tse_chunk} -o {fn_output_dir}/tse_native_chunk_{side}.nii.gz')
    if fn_mprage:
        c3d.execute(f'{fn_mprage} -o {fn_output_dir}/mprage.nii.gz.nii.gz')
        
    # Save the affine to template matrix
    fn_out_affine_to_template = f'{fn_output_dir}/affine_t1_to_template/t1_to_template_affine.mat'
    if fn_affine_to_template:
        shutil.copy(fn_affine_to_template, fn_out_affine_to_template)
    else:
        np.savetxt(fn_out_affine_to_template, np.eye(4))
        
    # Return the ASHS folder for this
    return ASHSFolder(fn_output_dir, side, 'bootstrap', correction_mode)
        
                      

    
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
        self.fit_lddmm_result_reduced = self.fn_fit('fitted_lddmm_template_reduced.vtk')
        self.fit_omt_match = self.fn_fit('fitted_omt_match.vtk')
        self.fit_omt_hw_target = self.fn_fit('fitted_omt_hw_target.vtk')
        self.fit_omt_match_to_hw = self.fn_fit('fitted_omt_match_to_hw.vtk')
        self.fit_dist_stat = self.fn_fit('fitted_dist_stat.json')

        self.thick_boundary = self.fn_thick('thick_boundary_raw.vtk')
        self.thick_boundary_sm = self.fn_thick('thick_boundary_smooth.vtk')
        self.thick_tetra_mesh = self.fn_thick('thickness_tetra.vtk')
        self.thick_skeleton = self.fn_thick('skeleton.vtk')
        self.thick_result = self.fn_thick('template_thickness.vtk')
        self.thick_roi_summary = self.fn_thick('thickness_roi_summary.csv')

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
        category_order: list, background_category, softmax_scale = 0):
    
    # Add up the posteriors corresponding to the different tissue labels
    p_cat = {}
    img_ref = None
    for cat in category_order:
        p_cat[cat] = None
        for v in category_labels[cat]:
            img_itk = ashs_posteriors.get(v, None)
            if img_itk:
                img_ref = img_itk
                img_arr = sitk.GetArrayFromImage(img_itk)
                p_cat[cat] = img_arr if p_cat[cat] is None else p_cat[cat] + img_arr

    # Normalize the posteriors so that the max total probability is one. Notably the posteriors
    # output by JLF can be negative and here we clip them to the zero/one range. Finally, we
    # optionally apply softmax
    x = np.concatenate([p_cat[cat][:, :, :, np.newaxis] for cat in category_order], axis=3)
    p_max = np.max(np.sum(x, 3))
    x = np.clip(x, 0, p_max) / p_max;

    # Allocate the missing probability to the background 
    bix = category_order.index(background_category)
    x[:,:,:,bix] += 1 - x.sum(axis=3)

    # Finally optional softmax
    if softmax_scale > 0:
        x = scipy.special.softmax(x * softmax_scale, 3)

    # Generate an ITK vector image of the tissue class probabilities
    img_cat = sitk.GetImageFromArray(x, True)
    img_cat.CopyInformation(img_ref)

    # Return the results
    return img_cat


# Routine that takes ASHS posteriors and combines them into a tissue label image
def ashs_posteriors_to_tissue_labels(
        ashs_posteriors: dict, category_labels: dict, 
        category_order: list, background_category, softmax_scale = 0):
    
    img_cat = ashs_posteriors_to_tissue_probabilities(
        ashs_posteriors, category_labels, category_order, background_category, softmax_scale)    
    tseg = sitk.GetImageFromArray(np.argmax(sitk.GetArrayFromImage(img_cat),3))
    tseg.CopyInformation(img_cat)
    lab_to_idx = { lab:i for (i, lab) in enumerate(category_order) }
    return tseg, lab_to_idx


# Routine that takes ASHS posteriors and combines them into a segmentation image
def ashs_posteriors_to_segmentation(ashs_posteriors: dict):
    c3d = Convert3D()
    reps = []
    for i, (label, p) in enumerate(ashs_posteriors.items()):
        if p is not None:
            c3d.push(p)
            reps.append(f'{i} {label}')
    print(" ".join(reps))
    c3d.execute(f'-vote -replace {" ".join(reps)}')
    return c3d.peek(-1)


# This routine takes per-label posteriors and a tissue class posterior and recomputes
# the per-label posteriors so that they add up to the tissue class posterior. This is
# done using fast marching
def reassign_label_posteriors_to_new_tissue_posterior(
        ashs_posteriors: dict, tissue_posterior: sitk.Image, labels: list):
    
    # Vote among the posteriors in this tissue class 
    c3d = Convert3D()
    c3d.add_image('TP', tissue_posterior)
    used_labels = []
    for label in labels:
        if label in ashs_posteriors:
            c3d.push(ashs_posteriors[label])
            used_labels.append(label)
    c3d.execute('-vote')

    # Now split the voting result and fast march for each label over the tissue prob
    c3d.execute(f'-split -foreach-comp {len(used_labels)} -push TP -thresh 0.5 inf 1 0 -times '
                f'-insert TP 1 -fast-marching 20 -reciprocal -endfor '
                f'-foreach -scale 10 -endfor -softmax -foreach -push TP -times -endfor')
    
    # Generate the updated posteriors
    return { l : c3d.peek(i) for (i, l) in enumerate(used_labels) }

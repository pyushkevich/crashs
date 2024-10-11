#!/usr/bin/env python3
import SimpleITK as sitk
import numpy as np
import argparse
from picsl_c3d import Convert3D
from picsl_greedy import Greedy3D
import crashs.upsample.upsample_net as upsample_net
from crashs.util import *
import os
import torch
import json


# Common inputs and outputs for the postprocessing task
class PreprocessWorkspace:
    def __init__(self, output_dir, id):
        self.output_dir = output_dir
        self.id = id
        self.fn_nnunet_input = f'{output_dir}/nnunet/input/MTL_000_0000.nii.gz'
        self.fn_nnunet_output = f'{output_dir}/nnunet/output/MTL_000.nii.gz'
        self.fn_upsample_wm_seg = f'{output_dir}/{id}_upsample_wmseg.nii.gz'
        self.dir_new_posteriors = f'{output_dir}/posteriors'


# Inputs and outputs for the T2 postprocessing task
class PreprocessT2Workspace(PreprocessWorkspace):
    def __init__(self, output_dir, id):
        PreprocessWorkspace.__init__(self, output_dir, id)
        self.fn_upsample_input = f'{output_dir}/{id}_chunk_gm_dg_seg.nii.gz'
        self.fn_upsample_output = f'{output_dir}/{id}_ivseg_unet_upsample.nii.gz'
        self.fn_upsample_output_bin = f'{output_dir}/{id}_ivseg_unet_upsample_bin.nii.gz'


def nnunet_wm_inference(template:Template, nnunet_model, ws: PreprocessWorkspace, device):

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from batchgenerators.utilities.file_and_folder_operations import load_json, join
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    # nnunet_model = f'{template.root}/nnunet/{upsample_opts["nnunet_wm"]["model"]}'
    predictor = nnUNetPredictor(verbose=True, device=device)
    use_folds = predictor.auto_detect_available_folds(nnunet_model, 'checkpoint_final.pth')

    dataset_json = load_json(join(nnunet_model, 'dataset.json'))
    plans = load_json(join(nnunet_model, 'plans.json'))
    plans_manager = PlansManager(plans)

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(nnunet_model, f'fold_{f}', 'checkpoint_final.pth'),
                                map_location=torch.device('cpu'))
        if i == 0:
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)

    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    network = nnUNetTrainer.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    )

    predictor.manual_initialization(
        network, plans_manager, configuration_manager, parameters,
        dataset_json, 'nnUNetTrainer',
        inference_allowed_mirroring_axes)

    #predictor.initialize_from_trained_model_folder(nnunet_model, None)
    predictor.predict_from_files(
        os.path.dirname(ws.fn_nnunet_input),
        os.path.dirname(ws.fn_nnunet_output),
        save_probabilities=False,
        overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)
    

def add_wm_segmentation_to_ashs_t2(
        ashs:ASHSFolder, template:Template, nnunet_model:str,  ws: PreprocessWorkspace, device):
    
    # Read the nnunet options from the preprocessing options
    with open(os.path.join(nnunet_model, 'config.json')) as config_json:
        nnunet_opts = json.load(config_json)

    # Read the options that affect upsampling
    target_orientation = np.array(nnunet_opts["target_orientation"])
    target_spacing = np.array(nnunet_opts["target_spacing"])

    # Next we need to generate the white matter segmentation. For this, we need the
    # whole-brain T1-weighted MRI. We extract the ROI around the segmentation and
    # set the orientation to match that of the nnU-Net training data
    c3d = Convert3D()
    c3d.execute(f'{ashs.mprage} -as T1  {ashs.tse_native_chunk} -thresh -inf inf 1 0 '
                f'-int 0 -reslice-matrix {ashs.affine_t1f_t2m} -trim 1vox '
                f'-push T1 -reslice-identity -swapdim {target_orientation}')

    # The T1 may need to be upsampled to target resolution before running nnU-Net. 
    # We can read from the upsample folder what the target image orientation and
    # target resolution are for the nnU-Net and then determine the upsampling
    # factors that match this resolution and orientation most closely
    source_spacing = np.array(c3d.peek(-1).GetSpacing())

    # Compute the scaling factors and perform upsampling
    upsample_factors = np.maximum(1, np.round(source_spacing / target_spacing))
    upsample_vec = 'x'.join([str(int(x)) for x in upsample_factors])
    print(f'Upsampling T1 ROI by {upsample_vec} using Non-Local Means')
    c3d.execute(f'-nlm-denoise -nlm-upsample {upsample_vec} -o {ws.fn_nnunet_input}')

    # Perform white matter segmentation 
    nnunet_inner_model = os.path.join(nnunet_model, nnunet_opts['model'])
    nnunet_wm_inference(template, nnunet_inner_model, ws, device)

    # Transform the white matter segmentation into T1 space
    tmppref = f'{ws.output_dir}/tmp/{ws.id}'
    g = Greedy3D()
    g.execute(f'-threads 1 -rf {ws.fn_upsample_output_bin} '
              f'-rm {ws.fn_nnunet_input} {tmppref}_t1sr_to_t2.nii.gz '
              f'-ri LABEL 0.2mm -rm {ws.fn_nnunet_output} {ws.fn_upsample_wm_seg} '
              f'-r {ashs.affine_t2f_t1m}')
    

def postprocess_t2_upsample(
        ashs:ASHSFolder, template:Template, upsample_opts:dict,  ws: PreprocessT2Workspace,
        bypass_wm = False):
    
    # Map labels to categories CA, SUB+ERC and Cortex
    all_lab = [ x for (k,v) in template.get_label_types().items() for x in v ]
    cat_map = { k : upsample_opts[f'{k}_labels'] for k in ['cortex', 'ca', 'suberc'] }
    fg_lab = [ x for (k,v) in cat_map.items() for x in v ]
    cat_map['bg'] = [ x for x in all_lab if x not in fg_lab ]

    # Group labels into these four categories
    img_hc_seg, lab_hc = ashs_posteriors_to_tissue_labels(
        ashs.posteriors, cat_map, ['bg','cortex', 'ca', 'suberc'], 'bg')

    # Split the upsampled image into GM and DG components
    c3d = Convert3D()
    c3d.execute(f'-mcs {ws.fn_upsample_output} {ws.fn_upsample_input} -thresh 1 1 1 0')
    img_up_gm, img_up_dg, img_gm_lores = c3d.peek(0), c3d.peek(1), c3d.peek(2);

    # Register the GM component to the in vivo GM
    g = Greedy3D()
    g.execute(
        "-a -dof 6 -n 100x100 -m SSD -ia-identity -threads 1 "
        "-i gm_lores gm_up -o affine",
        gm_lores = img_gm_lores, gm_up = img_up_gm, affine=None)
    g.execute(
        "-rf gm_up -rm gm_up gm_up_shift -rm dg_up dg_up_shift -r affine",
        dg_up=img_up_dg, gm_up_shift=None, dg_up_shift=None)
    img_up_gm, img_up_dg = g['gm_up_shift'], g['dg_up_shift']

    # Prefix for where to write temporary outouts
    tmppref = f'{ws.output_dir}/tmp/{ws.id}'

    sitk.WriteImage(img_up_gm, f'{tmppref}_img_up_gm.nii.gz')

    # Generate the band to add to white matter where hippocampus and cortex touch
    c3d.add_image('img_up_gm', img_up_gm)
    c3d.add_image('img_up_dg', img_up_dg)
    c3d.add_image('img_hc_seg', img_hc_seg)
    c3d.execute(f'-clear -push img_up_gm -thresh 0.7 inf 1 0 -as X -push img_hc_seg ' 
                f'-int 0 -reslice-identity -as Y '
                f'-thresh {lab_hc["cortex"]} {lab_hc["cortex"]} 1 0 -sdt '
                f'-push Y -thresh {lab_hc["ca"]} {lab_hc["ca"]} 1 0 -sdt '
                f'-push Y -thresh {lab_hc["suberc"]} {lab_hc["suberc"]} 1 0 -sdt '
                f'-foreach -scale -1 -endfor '
                f'-vote -shift 1 -push X -times -as Z -thresh 1 1 1 0 -as A -push Z -thresh 2 2 1 0 -as B '
                f'-dilate 1 1x1x1 -times -push A -dilate 1 1x1x1 -push B -times -add '
                f'-o {tmppref}_hc_overlap.nii.gz -as hc_overlap')
    
    # Reslice the WM segmentation into an upsampled T2 segmentation space and clean up,
    # getting rid of WM pixels that are too far from any GM, as well as any CSF stuck 
    # between GM and WM
    thresh_level = 0.5
    c3d.execute(f'-clear -push img_up_gm -thresh {thresh_level} inf 1 0 -as G '
                f'{ws.fn_upsample_wm_seg} -thresh 1 1 1 0 -smooth 0.2mm -reslice-identity -thresh 0.5 inf 1 0 -popas WG '
                f'-push G -stretch 0 1 1 0 -push WG -fm 5.0 -thresh 0 3 1 0 '
                f'-push G -stretch 0 1 1 0 -times -as W -push G -fm 20.0 -thresh 0 5 1 0 -push W -times '
                f'-o {tmppref}_wm_upsample_shift_pad_trim.nii.gz -as img_up_wm')

    # Split the gray matter probability map between CA and GM labels - get two probability maps
    c3d.execute(f'-clear -push img_hc_seg -replace {lab_hc["suberc"]} {lab_hc["cortex"]} '
                f'-replace 0 255 -split '
                f'-foreach -insert img_up_gm 1 -int 0 -reslice-identity -insert img_up_gm 1 -fm 5.0 -reciprocal -endfor '
                f'-pop -vote -split -foreach -push img_up_gm -times -endfor '
                f'-omc {tmppref}_gm_mtl_ca_prob.nii.gz -popas img_up_gm_pca -popas img_up_gm_pmtl')

    # Compute the fast marching distance from CA1 into CSF. We start with voxels that have CA1 probability L
    # and flow outwards
    c3d.execute(f'-clear -push img_up_gm -push img_up_dg -as D -max -as A '
                f'-stretch 0 1 1 0 -popas B '
                f'-push img_up_gm_pca -push img_up_gm_pmtl -push D '
                f'-foreach -thresh {thresh_level} inf 1 0 -insert B 1 -fm 3.0 -reciprocal -replace inf 0 -endfor '
                f'-omc {tmppref}_csf_dist_to_ca_mtl_dg.nii.gz '
                f'-popas img_csf_dist_dg -popas img_csf_dist_mtl -popas img_csf_dist_ca')

    # Select only those voxels where the inverse distance to CA is greater than to DG/GM by factor of three
    c3d.execute(f'-clear -push img_csf_dist_dg -scale 3.0 '
                f'-push img_csf_dist_ca -scale 1.0 -push img_csf_dist_mtl -scale 1.0 '
                f'-vote -shift 1 -as img_ca_dg_mtl_assgt '
                f'-o {tmppref}_global_ca_dg_mtl_assgt.nii.gz '
                f'-thresh 2 2 1 0 -dup -comp -thresh 1 1 1 0 -times '
                f'-push img_up_gm -push img_up_dg -max -stretch 0 1 1 0 -times '
                f'-o {tmppref}_csf_pseudo_wm_prob.nii.gz -as img_pseudo_wm_prob')

    # Generate the padded GM, WM and CSF probability images
    c3d.execute(f'-clear -push img_up_gm -pad 10x10x10 10x10x10 0 -as Q -thresh 0 {thresh_level} 1 0 -popas M '
                f'-push Q -stretch 0 {thresh_level} 0 0.5 -clip 0 0.5 -push M -times '
                f'-push Q -stretch {thresh_level} 1 0.5 1.0 -clip 0.5 1.0 -push M -stretch 0 1 1 0 -times -max '
                f'-as B -dup -push hc_overlap -int 0 -reslice-identity -as O1 '
                f'-push B -push img_pseudo_wm_prob -int 1 -reslice-identity -as O2 -max -as O '
                f'-stretch 0 1 1 0 -times -popas C '
                f'-push B -push C -push B -scale -1 -add -smooth 0.5mm -add -clip 0 1 '
                f'-as img_prob_gm -o {tmppref}_prob_gm.nii.gz -stretch 0 1 1 0 -as PNG '
                f'-push Q -push img_up_wm -reslice-identity -push O -max -as W '
                f'-push PNG -times -as img_prob_wm -o {tmppref}_prob_wm.nii.gz '
                f'-push PNG -push W -stretch 0 1 1 0 -times -as img_prob_csf -o {tmppref}_prob_csf.nii.gz ')
    
    # Generate the post-processed posteriors
    os.makedirs(f'{ws.dir_new_posteriors}', exist_ok=True)

    # We need to generate a new segmentation image with the white matter, i.e., we take the posteriors 
    # but take into account that some voxels are now given the WM label
    c3d.execute('-clear')
    label_order = []
    for label, posterior in ashs.posteriors.items():
        if label not in template.get_labels_for_tissue_class('wm'):
            c3d.push(posterior)
            label_order.append(label)
    label_order.append(template.get_labels_for_tissue_class('wm')[0])

    c3d.execute(f'-vote -popas S -push img_prob_wm -push S -int 0 -reslice-identity '
                f'-push img_prob_wm -thresh 0.5 inf 255 0 -max -split ')

    # Ok, now we have a segmentation on the stack where we have all the bg/csf labels 
    # in consecutive order and a white matter label as 255. We want to propagate each
    # of these images through the corresponding tissue probability map
    for label, posterior in ashs.posteriors.items():
        if label not in template.get_labels_for_tissue_class('wm'):
            cat_match = [ k for k in ['dg','cortex','suberc','ca'] if label in upsample_opts[f'{k}_labels'] ]
            cat = cat_match[0] if len(cat_match) > 0 else 'bg'
            pmap = 'img_prob_csf' if cat in ['bg','dg'] else 'img_prob_gm'
            c3d.execute(f'-push {pmap}')
    c3d.execute(f'-push img_prob_wm')
    
    # Compute softmax and an updated segmentation
    c3d.execute(f'-foreach-comp {len(label_order)} -as P -thresh 0.5 inf 1 0 -times -insert P 1 '
                f'-fast-marching 20 -reciprocal -endfor -foreach -scale 10 -endfor -softmax')
    
    # We can now output the new posteriors for CRASHS to use.
    upsampled_posterior_pattern = f'{ws.dir_new_posteriors}/preproc_posterior_%03d.nii.gz'
    for i, label in enumerate(label_order):
        sitk.WriteImage(c3d.peek(i), upsampled_posterior_pattern % (label,))

    # Also combine the posteriors into a segmentation 
    reps = ' '.join([ f'{i} {label}' for i, label in enumerate(label_order)])
    c3d.execute(f'-vote -replace {reps} -o {tmppref}_ivseg_ashs_upsample.nii.gz')
    
    # Finally, output the new posteriors
    return upsampled_posterior_pattern


def postprocess_t1_wm(
        ashs:ASHSFolder, template:Template, upsample_opts:dict, ws: PreprocessT2Workspace):
    
    # Prefix for where to write temporary outouts
    tmppref = f'{ws.output_dir}/tmp/{ws.id}'

    # TODO: It would be nice to add the strip of WM between hippocampus and MTL cortex as we do
    # for the T2 ASHS post-processing, but it is not clear how to do this because there are places
    # where MTL and hippocampus label are supposed to be adjacent in T1 ASHS. This stems from 
    # there not being a separate subiculum label in T1 ASHS. It may be possible to do something
    # based on the ranging of ERC but seems very complex.

    # Convert ASHS posteriors into background and GM posteriors
    tissue_cat = ['gm', 'bg']
    tissue_cat_labels = { k: template.get_labels_for_tissue_class(k) for k in tissue_cat }
    img_cat = ashs_posteriors_to_tissue_probabilities(ashs.posteriors, tissue_cat_labels, tissue_cat, 'bg')

    # Incorporate the WM segmentation into the ASHS segmentation space and clean up,
    # getting rid of WM pixels that are too far from any GM, as well as any CSF stuck 
    # between GM and WM
    thresh_level = 0.5
    c3d = Convert3D()
    c3d.add_image('img_prob_gm', sitk.VectorIndexSelectionCast(img_cat, 0))
    c3d.execute(f'-push img_prob_gm -o {tmppref}_prob_gm.nii.gz '
                f'{ws.fn_upsample_wm_seg} -thresh 1 1 1 0 -smooth 0.2mm -reslice-identity -thresh 0.5 inf 1 0 -popas WG '
                f'-push img_prob_gm -stretch 0 1 1 0 -push WG -fm 5.0 -thresh 0 5 1 0 '
                f'-push img_prob_gm -stretch 0 1 1 0 -times -as W -push img_prob_gm -fm 20.0 -thresh 0 5 1 0 -push W -times '
                f'-o {tmppref}_csf_pseudo_wm_prob.nii.gz -as img_pseudo_wm_prob')

    # Generate the new GM, WM, and CSF probability images
    c3d.execute(f'-clear -push img_prob_gm -dup -stretch 0 1 1 0 -popas PNG '
                f'-push img_pseudo_wm_prob -push PNG -times -as img_prob_wm -o {tmppref}_prob_wm.nii.gz '
                f'-push PNG -push W -stretch 0 1 1 0 -times -as img_prob_csf -o {tmppref}_prob_csf.nii.gz ')
    
    # Generate the updated posteriors for the background labels
    post_bg = reassign_label_posteriors_to_new_tissue_posterior(
        ashs.posteriors, c3d.peek(-1), tissue_cat_labels['bg'])
    
    # The gray matter posteriors stay the same
    post_gm = reassign_label_posteriors_to_new_tissue_posterior(
        ashs.posteriors, c3d.peek(-3), tissue_cat_labels['gm'])

    # The white matter posterior is what we generated
    post_wm = { template.get_labels_for_tissue_class('wm')[0] : c3d.peek(-2) }

    # All the posteriors together
    post_new = { **post_bg, **post_gm, **post_wm }
    
    # Generate the post-processed posteriors
    os.makedirs(f'{ws.dir_new_posteriors}', exist_ok=True)
    upsampled_posterior_pattern = f'{ws.dir_new_posteriors}/preproc_posterior_%03d.nii.gz'
    for label, post in post_new.items():
        sitk.WriteImage(post, upsampled_posterior_pattern % (label,))

    # Compute a segmentation from the posteriors
    seg_new = ashs_posteriors_to_segmentation(post_new)
    sitk.WriteImage(seg_new, f'{tmppref}_ashs_with_wm.nii.gz')

    return upsampled_posterior_pattern


def upsample_t2(cdr: CrashsDataRoot, ashs:ASHSFolder, preproc_opts: dict, ws: PreprocessT2Workspace):
    
    # Locate the upsampling model
    upsample_args = argparse.Namespace(
        train = cdr.find_model(preproc_opts['upsample_model']),
        output = ws.output_dir,
        t2_roi = ashs.tse_native_chunk,
        t2_seg = ws.fn_upsample_input, 
        id = ws.id)

    upsample_net.do_apply_single(upsample_args)


def import_ashs_t2(cdr: CrashsDataRoot, ashs:ASHSFolder, template:Template, output_dir, id, device):

    # Read the preprocessing/upsampling options
    preproc_opts = template.json['preprocessing']['t2_alveus_param']

    # Initialize the output folder (use namespace format)
    for subdir in ['','/tmp','/nnunet/input','/nnunet/output']:
        os.makedirs(f'{output_dir}{subdir}', exist_ok=True)

    # Relevant filenames
    ws = PreprocessT2Workspace(output_dir, id)

    # Before upsampling, we need to create an image with two labels, 1 for the gray
    # matter excluding dentate, and 2 for the dentate. 
    lab_dg = preproc_opts["dg_labels"]
    cl_old = { cat: template.get_labels_for_tissue_class(cat) for cat in ['bg','gm','wm'] }
    cl_new = {
        'bg': [ l for l in cl_old['bg'] + cl_old['wm'] if l not in lab_dg ],
        'gm': [ l for l in cl_old['gm'] if l not in lab_dg ],
        'dg': lab_dg
    }

    # Vote between posteriors based on category labels
    img_dggm_seg, _ = ashs_posteriors_to_tissue_labels(ashs.posteriors, cl_new, ['bg','gm','dg'], 'bg')
    sitk.WriteImage(img_dggm_seg, ws.fn_upsample_input)

    # The first step is to upsample the ASHS T2 segmentation using the deep learning
    # upsampling model. This will increase the z-resolution of the segmentation
    upsample_t2(cdr, ashs, preproc_opts, ws)

    # Add the white matter label
    nnunet_model = cdr.find_model(template.get_white_matter_nnunet_model())
    add_wm_segmentation_to_ashs_t2(ashs, template, nnunet_model, ws, device)

    # Next, we need to propagate the original ASHS labels into the upsampled segmentations
    # and extend the white matter over the alveus
    upsampled_posterior_pattern = postprocess_t2_upsample(ashs, template, preproc_opts, ws)
    return upsampled_posterior_pattern
    

def add_wm_to_ashs_t1(cdr: CrashsDataRoot, ashs:ASHSFolder, template:Template, output_dir, id, device):

    # Read the preprocessing/upsampling options
    preproc_opts = template.json['preprocessing']['t1_add_wm_param']
    
    # Initialize the output folder (use namespace format)
    for subdir in ['','/tmp','/nnunet/input','/nnunet/output']:
        os.makedirs(f'{output_dir}{subdir}', exist_ok=True)

    # Relevant filenames
    ws = PreprocessWorkspace(output_dir, id)

    # Locate the NNUnet model
    nnunet_model = cdr.find_model(template.get_white_matter_nnunet_model())

    # Read the nnunet options from the preprocessing options
    with open(os.path.join(nnunet_model, 'config.json')) as config_json:
        nnunet_opts = json.load(config_json)

    # Read the options that affect upsampling
    target_orientation = np.array(nnunet_opts["target_orientation"])
    target_spacing = np.array(nnunet_opts["target_spacing"])

    # We simply copy the T1-ASHS "tse" image (which is an upsampled T1-MRI) 
    # as the nnU-Net input.
    # TODO: we maybe should check if for some reason the input has not been upsampled
    # and if so, upsample it to match what NNUnet expects
    c3d = Convert3D()
    c3d.execute(f'{ashs.tse_native_chunk} -swapdim {target_orientation} -o {ws.fn_nnunet_input}')

    # Perform white matter segmentation 
    nnunet_inner_model = os.path.join(nnunet_model, nnunet_opts['model'])
    # nnunet_wm_inference(template, nnunet_inner_model, ws, device)

    # Copy the nnUnet output as our WM result
    c3d.execute(f'{ws.fn_nnunet_output} -o {ws.fn_upsample_wm_seg}')

    # Next, we need to propagate the original ASHS labels into the upsampled segmentations
    # and extend the white matter over the alveus
    upsampled_posterior_pattern = postprocess_t1_wm(ashs, template, preproc_opts, ws)
    return upsampled_posterior_pattern


def upsample_only_ashs_t2(cdr: CrashsDataRoot, ashs:ASHSFolder, template:Template, output_dir, id, device):

    # Read the preprocessing/upsampling options
    preproc_opts = template.json['preprocessing']['t2_alveus_param']

    # Initialize the output folder (use namespace format)
    for subdir in ['','/tmp']:
        os.makedirs(f'{output_dir}{subdir}', exist_ok=True)

    # Relevant filenames
    ws = PreprocessT2Workspace(output_dir, id)

    # Before upsampling, we need to create an image with two labels, 1 for the gray
    # matter excluding dentate, and 2 for the dentate. 
    lab_dg = preproc_opts["dg_labels"]
    cl_old = { cat: template.get_labels_for_tissue_class(cat) for cat in ['bg','gm','wm'] }
    cl_new = {
        'bg': [ l for l in cl_old['bg'] + cl_old['wm'] if l not in lab_dg ],
        'gm': [ l for l in cl_old['gm'] if l not in lab_dg ],
        'dg': lab_dg
    }

    # Vote between posteriors based on category labels
    img_dggm_seg, _ = ashs_posteriors_to_tissue_labels(ashs.posteriors, cl_new, ['bg','gm','dg'], 'bg')
    sitk.WriteImage(img_dggm_seg, ws.fn_upsample_input)

    # The first step is to upsample the ASHS T2 segmentation using the deep learning
    # upsampling model. This will increase the z-resolution of the segmentation
    upsample_t2(cdr, ashs, preproc_opts, ws)
    
    # Set white matter segmentation to zero
    c3d = Convert3D()
    c3d.execute(f'{ashs.tse_native_chunk} -scale 0 -type uchar -o {ws.fn_upsample_wm_seg}')

    # Next, we need to propagate the original ASHS labels into the upsampled segmentations
    # and extend the white matter over the alveus
    upsampled_posterior_pattern = postprocess_t2_upsample(ashs, template, preproc_opts, ws, bypass_wm=True)
    return upsampled_posterior_pattern
    


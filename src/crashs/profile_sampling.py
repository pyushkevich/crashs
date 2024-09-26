#!/usr/bin/env python3
import SimpleITK as sitk
import numpy as np
import os
import argparse
import json
import pathlib
import SimpleITK as sitk
import torch
from pykeops.torch import LazyTensor
from crashs.util import Workspace
from crashs.vtkutil import *

def do_sampling(args):

    # Label map - will be allocated once we know how many meshes
    label_map, mask_map = None, None

    # Load the reference mesh
    pd_ref = load_vtk(args.mesh)
    v_ref = vtk_get_points(pd_ref)

    # Dimensions of the sampling array
    n_vert, n_prof, n_cases, n_labels = v_ref.shape[0], None, len(args.subjects), len(args.labels)

    # Iterate over the cases
    for i, id in enumerate(args.subjects):

        # Create a workspace for this CRASHS case. 
        fn_crashs = args.crashs_pattern.format(id = id)
        ws = Workspace(fn_crashs, id, None)

        # Find the layer meshes
        prof = ws.fn_fit_profile_meshes()
        if len(prof) == 0 or n_prof is not None and len(prof) != n_prof:
            raise ValueError(f'Missing profiles or wrong number of profiles for case {id}')

        # Load the image to sample
        print(f'Sampling image {args.image_pattern.format(id = id)}')
        img = sitk.ReadImage(args.image_pattern.format(id = id), outputPixelType=sitk.sitkFloat32)
        
        # Load the mask, if provided. If not, the mask will be the whole image
        mask = None
        if args.mask_pattern:
            print(f'  reading mask from {args.mask_pattern.format(id = id)}')
            mask = sitk.ReadImage(args.mask_pattern.format(id = id), outputPixelType=sitk.sitkFloat32)
        
        # Set up the label map
        if label_map is None:
            n_prof = len(prof)
            label_map = np.zeros((n_vert, n_prof, n_cases, n_labels if mask else n_labels+1))
            mask_map = np.zeros((n_vert, n_prof, n_cases, 1))

        # Split the segmentation into binary images
        x_img = sitk.GetArrayFromImage(img)
        p_lab = np.stack([ x_img == l for l in args.labels]).astype(np.float32)
        
        # When there is no mask, add a background label for pixels that are not assigned any label
        if not mask:
            p_lab = np.append(p_lab, [1 - p_lab.sum(0)], axis=0)

        # Put it back into a vector ITK image (not very efficient)
        img_p = sitk.GetImageFromArray(p_lab.transpose(1,2,3,0), isVector=True)
        img_p.CopyInformation(img)

        # Smooth the image a bit
        if args.label_smooth > 0:
            img_p = sitk.SmoothingRecursiveGaussian(img_p, args.label_smooth)
            if mask:
                mask = sitk.SmoothingRecursiveGaussian(mask, args.label_smooth)

        # Sample from each layer
        for layer in range(n_prof):

            # Get the sampling vertices
            print(f'  sampling layer {ws.fn_fit_profile_mesh(layer)}')
            pd_layer = load_vtk(ws.fn_fit_profile_mesh(layer))
            v_ras = vtk_get_points(pd_layer)

            # Convert to ITK's LPI (not RAS) physical coordinates
            v_lpi = v_ras @ np.diag([-1, -1, 1]) 

            # Iterate over vertices - this is slow but whatever
            for j in range(len(v_lpi)):
                try:
                    l = img_p.EvaluateAtPhysicalPoint(v_lpi[j,:].tolist(), sitk.sitkLinear)
                    m = mask.EvaluateAtPhysicalPoint(v_lpi[j,:].tolist(), sitk.sitkLinear) if mask else 1.0
                    l = [ round(v, 5) for v in l ]
                    label_map[j, layer, i, :] = l
                    mask_map[j, layer, i, :] = m
                except:
                    pass
                
    # Average the label map between subjects
    mask_mean = np.mean(mask_map, 2)
    label_map_consensus = np.where(mask_mean > 0, np.mean(mask_map * label_map, 2) / mask_mean, np.nan)

    # Add as arrays to the target mesh
    for i in range(label_map_consensus.shape[2]):
        layer_prob = label_map_consensus[:,:,i].astype(np.float32)
        vtk_set_point_array(pd_ref, f'{args.array}_{i:03d}', layer_prob)
        
    # Add mask array
    if args.mask_array:
        vtk_set_point_array(pd_ref, f'{args.array}_mask', mask_mean[:,:,0].astype(np.float32))
        
    # If the user requested cell arrays, generate those too
    if args.cell:
        pd_ref = vtk_all_point_arrays_to_cell_arrays(pd_ref)
        cell_label_map_consensus = np.stack([
            vtk_get_cell_array(pd_ref, f'{args.array}_{i:03d}') for i in range(label_map_consensus.shape[2])  
        ], axis=2)
        vtk_set_cell_array(pd_ref, f'{args.array}_argmax', np.argmax(cell_label_map_consensus, axis=2).astype(np.float32))
        
    # Also compute the argmax (i.e., segmentation)
    vtk_set_point_array(pd_ref, f'{args.array}_argmax', np.argmax(label_map_consensus, axis=2).astype(np.float32))
    
    # Save the final mesh
    print(np.array(args.labels))
    vtk_set_field_data(pd_ref, f'{args.array}_labels', np.array(args.labels, dtype=np.int32))
    save_vtk(pd_ref, args.output, binary=True)


def laplacian_kernel(x, y, sigma=0.1):
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    return (-D_ij.sqrt() / sigma).exp()


def do_mapping(args):

    # Load the template
    pd_temp = load_vtk(args.template)

    # Find the layer profile meshes
    ws = Workspace(args.crashs_dir, args.subject, None)
    prof = ws.fn_fit_profile_meshes()
    if len(prof) == 0:
        raise ValueError(f'Missing profiles or wrong number of profiles')
    
    # Stack all the profiles into a single list of vertices for RBF interpolation
    x = np.concatenate([
        vtk_get_points(load_vtk(ws.fn_fit_profile_mesh(i))) for i in range(len(prof)) ])

    # If label mode, get the label ids and label probabilities
    if args.labels:
        labels = vtk_get_field_data(pd_temp, f'{args.array}_labels')
        label_arr = [vtk_get_point_array(pd_temp, f'{args.array}_{i:03d}') for i in range(len(labels)+1)]
        label_map = np.stack(label_arr, axis = 2)
        
        # Organize the labels in the same order as the vertices
        b = label_map.transpose(1,0,2).reshape(-1, label_map.shape[2])

    # Load the reference image
    img = sitk.ReadImage(args.image, outputPixelType=sitk.sitkFloat32)

    # Extract pixels of interest
    x_img = sitk.GetArrayFromImage(img)

    # Limit to only requested labels if needed
    if args.target_labels:
        x_img = np.where(np.isin(x_img, args.target_labels), 1, 0)

    # Get the non-zero pixel indices (ITK ordering)
    nz = np.flip(np.transpose(np.stack(np.nonzero(x_img)).astype(np.float32)), 1)
    
    # Convert to RAS physical coordinates
    y = np.array([
        img.TransformContinuousIndexToPhysicalPoint(nz[j,:].tolist()) for j in range(nz.shape[0])])
    y = y @ np.diag([-1, -1, 1])
    
    # Perform RBF interpolation, so easy!
    K_yx = laplacian_kernel(
        torch.tensor(y, dtype=torch.float32), 
        torch.tensor(x, dtype=torch.float32),
        sigma=args.rbf_kernel)
    b_y = (K_yx @ torch.tensor(b, dtype=torch.float32)).detach().cpu().numpy()

    # Assign the values to the vertices
    l_best = np.argmax(b_y[:,:-1], 1)
    l_best_remap = np.zeros_like(l_best)
    for k, label in enumerate(labels):
        l_best_remap[l_best == k] = label
    x_img[x_img != 0] = l_best_remap

    # Save the image
    img_result = sitk.GetImageFromArray(x_img)
    img_result.CopyInformation(img)
    sitk.WriteImage(img_result, args.output)


class ProfileSamplingLauncher:

    def __init__(self, parse):

        parse.add_argument('-c','--crashs-pattern', type=str, required=True,
                           help='Pattern for input CRASHS directories ("{id}" replaced by subject id)')
        parse.add_argument('-i','--image-pattern', type=str, required=True,
                           help='Pattern for input images to sample')
        parse.add_argument('-M','--mask-pattern', type=str, required=False,
                           help='Pattern of mask images to sample (optional)')        
        parse.add_argument('-s','--subjects', type=str, required=True, nargs='+',
                           help='IDs of subjects to sample')
        parse.add_argument('-a','--array', type=str, required=True, 
                           help='Name of array where to store the sampled values')
        parse.add_argument('-A','--mask-array', type=str, required=False, 
                           help='Name of array where to store the sampled mask values')
        parse.add_argument('-m','--mesh', type=str, required=True, 
                           help='Reference mesh to which the sampled values will be added')
        parse.add_argument('-o','--output', type=str, required=True, 
                           help='Output filename for the saved mesh')
        parse.add_argument('-l','--labels', type=int, nargs='+',
                           help='Sample specified labels from a multi-label image')
        parse.add_argument('-S', '--label-smooth', '--smooth', type=float, default=0.4,
                           help='Amount of smoothing to apply when sampling labels/data (mm)')
        parse.add_argument('-C', '--cell', action='store_true', 
                           help='Generate cell arrays in addition to the point arrays')
        
        parse.set_defaults(func = lambda args : self.run(args))
        
    def run(self, args):
        do_sampling(args)
        
        
class ProfileMappingLauncher:
    
    def __init__(self, parse):

        parse.add_argument('-c','--crashs-dir', type=str, required=True,
                           help='CRASHS output directory for subject we want to map to')
        parse.add_argument('-i','--image', type=str, required=True,
                           help='Reference image to which values should be mapped.')
        parse.add_argument('-s','--subject', type=str, required=True,
                           help='IDs of CRASHS subject')
        parse.add_argument('-a','--array', type=str, required=True, 
                           help='Name of array that contains data to be mapped')
        parse.add_argument('-t','--template', type=str, required=True, 
                           help='Template mesh that stores the sampled values')
        parse.add_argument('-o','--output', type=str, required=True, 
                           help='Output filename for the saved image')
        parse.add_argument('-l','--labels', action='store_true', 
                           help='Multi-label mode (see sample command)')
        parse.add_argument('-k', '--rbf-kernel', type=float, default=0.4,
                           help='Radial basis function kernel size (mm)')
        parse.add_argument('-T', '--target-labels', type=int, nargs='+',
                           help='List of labels that should be replaced in the reference image. '
                                'By default, all non-zero labels are replaced.')
        parse.set_defaults(func = lambda args : self.run(args))
        
    def run(self, args):
        do_mapping(args)
    

if __name__ == '__main__':

    # Create a parser with subparsers f
    parse = argparse.ArgumentParser(description="CRASHS profile sampling tool")
    subparsers = parse.add_subparsers(help='sub-command help')

    # Set up the parser for sampling from a collection of CRASHS directories
    p_sample = subparsers.add_parser('sample', help='Collect samples from one or more individuals')
    p_sample.set_defaults(func=do_sampling)
    
    # Set up a parser for labeling an image in subject space
    p_map = subparsers.add_parser('map', help='Map previously sampled values into subject space')
    p_map.set_defaults(func=do_mapping)
    
    args = parse.parse_args()
    args.func(args)
    


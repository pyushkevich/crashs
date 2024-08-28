import numpy as np
import torch
import torch.nn.functional as F
import monai
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import glob
import os
import json
import argparse
import sys

# Load my device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Load an image using SITK and return the tensor and the NIFTI transform
def get_sform(img):
    D = np.array(img.GetDirection()).reshape(3,3)
    s = np.array(img.GetSpacing())
    org = np.array(img.GetOrigin())
    M = np.eye(4)
    M[:3,:3] = D @ np.diag(s)
    M[:3,3] = org
    return M


# Load an image
def read_sitk(fname):
    img = sitk.ReadImage(fname)
    aff = get_sform(img)
    T = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32).to(device)[None,None,:]
    return img, aff, T


def rodrigues(v, theta):
    K = torch.zeros((v.shape[0],3,3), dtype=v.dtype, device=v.device)
    K[:,0,1], K[:,1,0] = -v[:,2], v[:,2]
    K[:,2,0], K[:,0,2] = -v[:,1], v[:,1]
    K[:,1,2], K[:,2,1] = -v[:,0], v[:,0]
    return torch.eye(3, dtype=v.dtype, device=v.device)[None,:,:] + torch.sin(theta)[:,None,None] * K + (1 - torch.cos(theta))[:,None,None] * (K @ K)


def get_torch_to_vox_matrix(T_img):
    img_pt2vox = torch.eye(4, device=T_img.device, dtype=torch.float32)
    s = torch.flip(torch.tensor(T_img.shape[-3:], device=T_img.device, dtype=torch.float32),[0]) - 1.0
    img_pt2vox[:3,:3] = torch.diag(s / 2.)
    img_pt2vox[:3,3] = s / 2.
    return img_pt2vox


class PatchRotationSampler:

    def __init__(self, T_img, T_seg, patch_size, rot_sigma, align_corners=True):
        self.T_img = T_img
        self.T_seg = T_seg
        self.device = T_img.device
        self.patch_size = patch_size
        self.T_img_size = torch.flip(torch.tensor(T_img.shape[-3:], device=T_img.device, dtype=torch.float32),[0])
        self.T_patch_size = torch.flip(torch.tensor(patch_size, device=T_img.device, dtype=torch.float32),[0])
        self.rot_sigma = torch.flip(torch.tensor(rot_sigma, dtype=torch.float32, device=device),[0])

        # Define the mapping from the torch grid for the whole image to the voxel index
        self.img_pt2vox = torch.eye(4, device=T_img.device, dtype=torch.float32)
        self.img_pt2vox[:3,:3] = torch.diag((self.T_img_size-1) / 2.)
        self.img_pt2vox[:3,3] = (self.T_img_size-1) / 2.

        # Define the inverse of this mapping
        self.img_vox2pt = torch.linalg.inv(self.img_pt2vox)

        # Define the mapping from the torch grid for the patch to the center of the image
        self.patch_pt2vox = torch.eye(4, device=T_img.device, dtype=torch.float32)
        self.patch_pt2vox[:3,:3] = torch.diag((self.T_patch_size - 1) / 2.)
        self.patch_pt2vox[:3,3] = (self.T_img_size - 1) / 2.

        # The rectangle where the center of the patch may be located
        self.sr_scale = self.T_img_size - self.T_patch_size
        self.sr_center = (self.T_img_size - 1) / 2.

        # Get the list of voxel locations inside the segmentation and inside the image
        self.nz = torch.flip(torch.nonzero(T_seg.sum(1)[0,:]), [1]) - self.sr_center[None,:]
        self.nz = self.nz[torch.all(torch.abs(self.nz) < self.sr_scale[None,:]/2,1),:]

        # Save the align corners property
        self.align_corners = align_corners

    def sample(self, k, with_rotation=True):

        # Draw the next random sample from the iterator
        sample = torch.randint(0, self.nz.shape[0], (k,))
        sample_offset = self.nz[sample,:]
        self.sample_offset = sample_offset

        # Turn this into a transform
        sample_tform = torch.eye(4, device=self.device, dtype=torch.float32)[None,:,:].repeat(k,1,1)
        if with_rotation:

            # Compute a batch of rotation matrices in voxel space
            v_theta = torch.normal(torch.zeros(k,1,device=self.device, dtype=torch.float32), self.rot_sigma)
            theta = torch.norm(v_theta, dim = 1)
            v = v_theta / (theta[:,None] + 1e-6)
            R = rodrigues(v, theta)
            b = torch.matmul(R, self.sr_center[None,:,None]).squeeze()
            b = sample_offset + self.sr_center[None,:] - torch.matmul(R, self.sr_center[None,:,None]).squeeze()

            # Make the rotations be around the center
            sample_tform[:,0:3,0:3] = R
            sample_tform[:,0:3,3] = b
            
        else:            
            sample_tform[:,0:3,3] = sample_offset

        # Compute the mapping that takes the patch to the pixel space, applies offset, then
        # takes back to the image torch coordinates
        pt_tform = torch.matmul(self.img_vox2pt[None,:,:],torch.matmul(sample_tform, self.patch_pt2vox[None,:,:]))
        self.pt_tform = pt_tform
        grid = F.affine_grid(pt_tform[:,0:3,:], (k,1) + self.patch_size, align_corners=self.align_corners)
        return grid, F.grid_sample(self.T_img.expand(k,-1,-1,-1,-1), grid, align_corners=self.align_corners)
    
    def sample_other(self, img, grid):
        return F.grid_sample(img.expand(grid.shape[0],-1,-1,-1,-1), grid, align_corners=self.align_corners)

    
class PatchRotationSampleMapper:

    def __init__(self, T_ref, T_mov, A_ref, A_mov, upsample=None):

        # Compute the transform from fixed to moving
        Q_ref = torch.tensor(A_ref, dtype=T_ref.dtype, device=T_ref.device) @ get_torch_to_vox_matrix(T_ref)
        Q_mov = torch.tensor(A_mov, dtype=T_ref.dtype, device=T_ref.device) @ get_torch_to_vox_matrix(T_mov)
        self.Q = torch.linalg.inv(Q_mov) @ Q_ref

        # Store the images
        self.T_ref = T_ref
        self.T_mov = T_mov
        self.upsample = upsample

    def to_moving(self, grid):
        grid_mov = torch.einsum('ij,bxyzj->bxyzi', self.Q[:3,:3], grid) + self.Q[:3,3]
        if self.upsample is not None:
            sz_out = tuple( int(self.upsample[i] * x) for i,x in enumerate(grid_mov.shape[1:4]) )
            grid_mov = F.interpolate(grid_mov.permute(0,4,1,2,3), sz_out, mode='trilinear', align_corners=True).permute(0,2,3,4,1)
        return grid_mov


# This class stores the data for one subject
class SubjectData:

    def __init__(self, id, patch_size=(5,32,32), upsample=None):
        self.id = id

        # Read the low-resolution in vivo image and segmentation
        self.img_a, self.aff_a, self.T_a = read_sitk(f'input/{id}/{id}_t2_roi.nii.gz')
        self.seg_a, _, self.T_sa = read_sitk(f'input/{id}/{id}_ivseg.nii.gz')

        # Generate one-hot encoding of the labels
        self.T_oha = torch.cat(tuple((torch.where(self.T_sa==lab, 1., 0.) for lab in (1,2))), 1)

        # Also read the high-resolution data
        self.img_b, self.aff_b, self.T_b = read_sitk(f'input/{id}/{id}_xv_to_iv_t2_hires_warped_1x1x5.nii.gz')
        self.seg_b, _, self.T_sb = read_sitk(f'input/{id}/{id}_merge_hisseg_srlmsplit_warped_to_iv_1x1x5.nii.gz')
        self.msk_b, _, self.T_mb = read_sitk(f'input/{id}/{id}_merge_bmask_warped_to_iv_1x1x5.nii.gz')

        # Isolate the gray matter label
        self.T_ohb = torch.cat(tuple((torch.where(self.T_sb==lab, 1., 0.) for lab in (1,2))), 1)

        # Create the samplers
        self.prs = PatchRotationSampler(self.T_a, self.T_oha, patch_size, (10,0,0))

        # Upsampling
        if upsample is None:
            upsample = np.flip(np.round(np.array(self.img_a.GetSpacing()) / np.array(self.img_b.GetSpacing())))
        self.prsm = PatchRotationSampleMapper(self.T_a, self.T_b, self.aff_a, self.aff_b, upsample)
        
    def sample(self, k, with_rotation=True):
        grid, T_sam = self.prs.sample(k, with_rotation=with_rotation)
        T_sam_seg = self.prs.sample_other(self.T_oha, grid)

        grid_hr = self.prsm.to_moving(grid)
        T_sam_hr = F.grid_sample(self.T_b.expand(k,-1,-1,-1,-1), grid_hr, align_corners=True)
        T_sam_seg_hr = F.grid_sample(self.T_ohb.expand(k,-1,-1,-1,-1), grid_hr, align_corners=True)
        T_sam_msk_hr = F.grid_sample(self.T_mb.expand(k,-1,-1,-1,-1), grid_hr, align_corners=True)

        return T_sam, T_sam_seg, T_sam_hr, T_sam_seg_hr, T_sam_msk_hr
    
    def sample_a_only(self, k, with_rotation=True):
        grid, T_sam = self.prs.sample(k, with_rotation=with_rotation)
        T_sam_seg = self.prs.sample_other(self.T_oha, grid)
        return T_sam, T_sam_seg

    def sample_b_only(self, k, with_rotation=True):
        grid, _ = self.prs.sample(k, with_rotation=with_rotation)
        grid_hr = self.prsm.to_moving(grid)
        T_sam_hr = F.grid_sample(self.T_b.expand(k,-1,-1,-1,-1), grid_hr, align_corners=True)
        T_sam_seg_hr = F.grid_sample(self.T_ohb.expand(k,-1,-1,-1,-1), grid_hr, align_corners=True)
        T_sam_msk_hr = F.grid_sample(self.T_mb.expand(k,-1,-1,-1,-1), grid_hr, align_corners=True)
        return T_sam_hr, T_sam_seg_hr, T_sam_msk_hr


# Because of anisotropy, I think it just makes sense to treat slices that we are upsampling
# as channels, at least to begin with. Later we'll see. 
class UpsampleNet(torch.nn.Module):

    def __init__(self, in_slices, upsample, n_labels=2):
        super(UpsampleNet, self).__init__()

        # Create a UNet that takes in an image and the segmentation as channels
        self.upsample = upsample
        self.n_labels = n_labels
        self.unet = monai.networks.nets.UNet(
            spatial_dims = 2,
            in_channels = in_slices * (n_labels+1),
            out_channels = in_slices * n_labels * upsample,
            channels = tuple(upsample * k for k in (16, 32, 64, 128)),
            strides=(2, 2, 2),
            num_res_units=2)
                
    def forward(self, x_img, x_seg):
        # Get input dimensions
        b,_,d,h,w = x_img.shape
        k = self.upsample
        l = self.n_labels

        # Combine the image and the segmentation
        x = torch.cat((x_img, x_seg), 1).flatten(1,2)
        
        # Pass the data through the u-net
        y = self.unet(x)

        # Split into slices/channels and apply softmax for each original slice
        # z=torch.softmax(y.reshape([b,k,d,h,w]), 1) * x.reshape(b,2,d,h,w)[:,1:,:,:,:].repeat(1,k,1,1,1)
        z = y.reshape([b,k * l,d,h,w])

        # Reshape into a single-channel 3D image
        z = z.permute(0,2,1,3,4).reshape(b,l,-1,h,w)

        return z


# Code to plot a sample
def plot_sample(T_sam, T_sam_seg, T_sam_hr, T_sam_seg_hr, T_sam_msk_hr):
    n_sam = T_sam.shape[0]
    fig,ax = plt.subplots(2,T_sam.shape[0], figsize=(3*T_sam.shape[0],6))
    for k in range(n_sam):
        ax[0,k].imshow(T_sam[k,0,2,:,:].detach().cpu().numpy(), cmap='gray')
        ax[0,k].contour(T_sam_seg[k,0,2,:,:].detach().cpu().numpy(), [0.5], colors='red', alpha=0.65)
        ax[0,k].contour(T_sam_seg[k,1,2,:,:].detach().cpu().numpy(), [0.5], colors='yellow', alpha=0.65)
        mid = T_sam_hr.shape[2]//2
        ax[1,k].imshow(T_sam_hr[k,0,mid,:,:].detach().cpu().numpy(), cmap='gray')
        for z in range(-2, 3):
            ax[1,k].contour(T_sam_seg_hr[k,0,mid+z,:,:].detach().cpu().numpy(), [0.5], colors='red', alpha=0.2)
            ax[1,k].contour(T_sam_seg_hr[k,1,mid+z,:,:].detach().cpu().numpy(), [0.5], colors='yellow', alpha=0.2)
        ax[1,k].imshow(T_sam_msk_hr[k,0,T_sam_msk_hr.shape[2]//2,:,:].detach().cpu().numpy(), alpha=0.4, vmin=0, vmax=1)

    return fig, ax

# Parse the manifest to get the train and test ids
def read_train_test(args):
    d = json.load(args.manifest)
    id_all = d['cases']
    id_test = d['folds'][args.fold]
    id_train = [ id for id in id_all if not id in id_test ]
    print('Training cases:', id_train)
    print('Test cases    :', id_test)
    return id_train, id_test


def do_train(args):
    print('hello')

    # Read the train/test split
    id_train, id_test = read_train_test(args)

    # Read the config file
    config = json.load(args.config)

    # Create the subject data
    patch_size = tuple(config.get('patch_size', [5,32,32]))
    upsample_factor = tuple(config.get('upsample_factor', [5,1,1]))
    lr = config.get('lr', 1e-4)
    w_target = config.get('w_target', 2.0) 

    # Load the individual cases
    d_train = { k: SubjectData(k, patch_size, upsample=upsample_factor) for k in id_train }

    # Create the upsample network and losses
    mynet = UpsampleNet(patch_size[0], upsample_factor[0]).to(device)
    loss_target = monai.losses.MaskedDiceLoss(sigmoid=True)
    loss_consist = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(mynet.parameters(), lr)

    # Do the training
    n_sam = args.batch_size
    for epoch in range(args.epochs):
        ef = { k: 0.0 for k in ('target', 'consist', 'total') }
        for id, sd in d_train.items():
            T_sam, T_sam_seg, T_sam_hr, T_sam_seg_hr, T_sam_msk_hr = sd.sample(n_sam)

            # Perform forward pass
            optimizer.zero_grad()
            T_sam_up = mynet.forward(T_sam, T_sam_seg)

            # Compute Dice with target shape
            f_target = loss_target(T_sam_up, T_sam_seg_hr, mask=T_sam_msk_hr)

            # Downsample back and compute Dice with itself
            T_sam_recon = F.avg_pool3d(T_sam_up, upsample_factor)
            f_consist = loss_consist(T_sam_recon, T_sam_seg)

            # Compute total loss
            f = f_target * w_target + f_consist

            # Perform optimizer step
            f.backward()
            optimizer.step()

            # Update the losses
            ef['target'] += f_target.item()
            ef['consist'] += f_consist.item()
            ef['total'] += f.item()

        ef = { k : v / len(d_train) for k,v in ef.items() }
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch:04d}: ' + ' '.join([f'{k}: {v:6.4f}' for k,v in ef.items()]))

    # Save the model description and state
    out_base = args.output
    os.makedirs(out_base, exist_ok=True)

    # Save the state dict
    torch.save(mynet.state_dict(), os.path.join(out_base,'model.dat'))

    # Save the current configuration
    model_desc = {'config': config, 
                  'id_train': id_train, 
                  'epochs': args.epochs, 
                  'batch_size': args.batch_size}
    with open(os.path.join(out_base,'config.json'),'wt') as fd:
        json.dump(model_desc, fd)


def do_test(args):

    # Read the train/test split
    id_train, id_test = read_train_test(args)

    # Load the model parameters
    with open(os.path.join(args.train, 'config.json'), 'rt') as fd:
        model_desc = json.load(fd)

    # Read specific parameters
    config = model_desc['config']
    patch_size = tuple(config.get('patch_size', [5,32,32]))
    upsample_factor = tuple(config.get('upsample_factor', [5,1,1]))

    # Instantiate and load the model
    mynet = UpsampleNet(patch_size[0], upsample_factor[0]).to(device)
    mynet.load_state_dict(torch.load(os.path.join(args.train, 'model.dat'), weights_only=True))
    mynet.eval()

    # Create the sliding window inference object
    inferer = monai.inferers.SlidingWindowInferer(patch_size, overlap=0.8)
    infer_model = lambda x: torch.sigmoid(mynet(x[:,0:1,:,:,:], x[:,1:,:,:,:]))

    # The model should be evalauated for both test and train subsets
    targets = { 'train': id_train, 'test': id_test } if args.no_train is False else { 'test': id_test }
    for target, target_ids in targets.items():
        save_dir = os.path.join(args.output, target)
        os.makedirs(save_dir, exist_ok=True)
        for id in target_ids:
            # Load the image and segmentation
            img_a, aff_a, T_a = read_sitk(f'{args.input_dir}/{id}/{id}_t2_roi.nii.gz')
            seg_a, _, T_sa = read_sitk(f'{args.input_dir}/{id}/{id}_ivseg.nii.gz')    
            T_oha = torch.cat(tuple((torch.where(T_sa==lab, 1., 0.) for lab in (1,2))), 1)

            # Perform upsampling with network
            print(f'Performing inference on {target} case {id}')
            with torch.no_grad():
                T_pred = inferer(inputs=torch.cat((T_a, T_oha), 1), network=infer_model)    

            # Save the upsampled image
            img_pred = sitk.GetImageFromArray(T_pred.squeeze().permute(1,2,3,0).detach().cpu().numpy(), isVector=True)
            
            # Compute the origin, spacing and direction
            spc, org = np.array(seg_a.GetSpacing()), np.array(seg_a.GetOrigin())
            spc_up = spc * np.array([1,1,0.2])
            org_up = org + 0.5 * (spc_up - spc)
            img_pred.SetDirection(seg_a.GetDirection())
            img_pred.SetOrigin(org_up)
            img_pred.SetSpacing(spc_up)

            # Save the image and binarize
            sitk.WriteImage(img_pred, f'{save_dir}/{id}_ivseg_unet_upsample.nii.gz')
            
            # Save the segmentation
            T_pred_lab = torch.where(torch.max(T_pred, 1).values > 0.5, torch.argmax(T_pred, 1) + 1, 0)
            img_pred_lab = sitk.GetImageFromArray(T_pred_lab.squeeze().detach().cpu().numpy())
            img_pred_lab.CopyInformation(img_pred)
            sitk.WriteImage(img_pred_lab, f'{save_dir}/{id}_ivseg_unet_upsample_bin.nii.gz')


# Apply to a single input dataset
def do_apply_single(args):

    # Load the model parameters
    with open(os.path.join(args.train, 'config.json'), 'rt') as fd:
        model_desc = json.load(fd)

    # Read specific parameters
    config = model_desc['config']
    patch_size = tuple(config.get('patch_size', [5,32,32]))
    upsample_factor = tuple(config.get('upsample_factor', [5,1,1]))

    # Instantiate and load the model
    mynet = UpsampleNet(patch_size[0], upsample_factor[0]).to(device)
    mynet.load_state_dict(torch.load(os.path.join(args.train, 'model.dat'), device, weights_only=True))
    mynet.eval()

    # Create the sliding window inference object
    inferer = monai.inferers.SlidingWindowInferer(patch_size, overlap=0.8)
    infer_model = lambda x: torch.sigmoid(mynet(x[:,0:1,:,:,:], x[:,1:,:,:,:]))

    # The model should be evalauated for both test and train subsets
    id, save_dir = args.id, args.output
    os.makedirs(save_dir, exist_ok=True)

    # Load the image and segmentation
    img_a, aff_a, T_a = read_sitk(args.t2_roi)
    seg_a, _, T_sa = read_sitk(args.t2_seg)    
    T_oha = torch.cat(tuple((torch.where(T_sa==lab, 1., 0.) for lab in (1,2))), 1)

    # Perform upsampling with network
    print(f'Performing inference on case {id}')
    with torch.no_grad():
        T_pred = inferer(inputs=torch.cat((T_a, T_oha), 1), network=infer_model)    

    # Save the upsampled image
    img_pred = sitk.GetImageFromArray(T_pred.squeeze().permute(1,2,3,0).detach().cpu().numpy(), isVector=True)
    
    # Compute the origin, spacing and direction
    spc, org = np.array(seg_a.GetSpacing()), np.array(seg_a.GetOrigin())
    spc_up = spc * np.array([1,1,0.2])
    org_up = org + 0.5 * (spc_up - spc)
    img_pred.SetDirection(seg_a.GetDirection())
    img_pred.SetOrigin(org_up)
    img_pred.SetSpacing(spc_up)

    # Save the image and binarize
    sitk.WriteImage(img_pred, f'{save_dir}/{id}_ivseg_unet_upsample.nii.gz')
    
    # Save the segmentation
    T_pred_lab = torch.where(torch.max(T_pred, 1).values > 0.5, torch.argmax(T_pred, 1) + 1, 0)
    img_pred_lab = sitk.GetImageFromArray(T_pred_lab.squeeze().detach().cpu().numpy())
    img_pred_lab.CopyInformation(img_pred)
    sitk.WriteImage(img_pred_lab, f'{save_dir}/{id}_ivseg_unet_upsample_bin.nii.gz')


def main():

    # Set up an argument parser
    parser = argparse.ArgumentParser(prog='upsample_net')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.add_argument('-m','--manifest', type=argparse.FileType('r'), required=True)
    parser_train.add_argument('-f','--fold', type=int, required=True)
    parser_train.add_argument('-c','--config', type=argparse.FileType('r'), required=True)
    parser_train.add_argument('-o','--output', type=str, required=True)
    parser_train.add_argument('--epochs', type=int, default=1000)
    parser_train.add_argument('--batch-size', type=int, default=64)
    parser_train.set_defaults(func=do_train)

    parser_test = subparsers.add_parser('test', help='train help')
    parser_test.add_argument('-m','--manifest', type=argparse.FileType('r'), required=True)
    parser_test.add_argument('-f','--fold', type=int, required=True)
    parser_test.add_argument('-t','--train', type=str, required=True, help='Folder containing trained model')
    parser_test.add_argument('-o','--output', type=str, required=True)
    parser_test.add_argument('--no-train', action='store_true', help='Only run on test data')
    parser_test.add_argument('-i','--input-dir', type=str, default='input', help='Base directory for input files')
    parser_test.set_defaults(func=do_test)

    parser_test = subparsers.add_parser('apply', help='apply help')
    parser_test.add_argument('-t','--train', type=str, required=True, help='Folder containing trained model')
    parser_test.add_argument('-o','--output', type=str, required=True, help='Output directory to save files')
    parser_test.add_argument('-g','--t2-roi', type=str, required=True, help='Input T2-MRI ROI')
    parser_test.add_argument('-s','--t2-seg', type=str, required=True, help='Input T2-MRI segmentation')
    parser_test.add_argument('-i','--id', type=str, required=True, help='Subject ID')
    parser_test.set_defaults(func=do_apply_single)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
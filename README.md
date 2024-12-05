# CRASHS: Cortical Reconstruction for Automatic Segmentation of Hippocampal Subfields (ASHS)
CRASHS is a surface-based modeling and groupwise registration pipeline for the human medial temporal lobe (MTL). It can be used to perform groupwise analysis of pointwise measures in the MTL, such as cortical thickness, longitudinal volume change, functional MRI activation, microstructure, etc. It uses similar principles to whole-brain surface-based analysis pipelines like [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) and [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043), but restricted to the MTL region. CRASHS is used to postprocess the results of [ASHS](https://github.com/pyushkevich/ashs) segmentation with certain ASHS atlases.

Some of the newer ASHS atlases include the white matter label, which is used by CRASHS. For other ASHS atlases, CRASHS can paint in the white matter label using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). CRASHS uses the [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043) technique implemented in the [NighRes software](https://nighres.readthedocs.io/en/latest/) to fit the white matter segmentation with a surface of spherical topology, and find a series of surfaces spanning between the gray/white boundary and the pial surface. The middle surface is inflated and registered to a population template, allowing surface-based analysis of MTL cortical thickness and other measures such as functional MRI and diffusion MRI. 

The CRASHS pipeline is described in the supplemental material to our paper in the special issue of Alzheimer's and Dementia on the [20th anniversary of ADNI](https://doi.org/10.1002/alz.14161).

## Installation using `pip`

CRASHS requires the `nighres` package, which cannot be installed with `pip`. To install `nighres`, please follow the [installation instructions](https://nighres.readthedocs.io/en/latest/). To our knowledge, the ARM64 architecture (newer Macs) is currently not supported.

Once `nighres` is installed, you can install CRASHS:

```sh
pip install crashs
python3 -m crashs fit --help
```

Or, if you want to use the latest development code and install in "editable" mode:

```sh
git clone https://github.com/pyushkevich/crashs
pip install -e ./crashs
```

## Installation using Docker
The CRASHS Docker container is available on DockerHub as `pyushkevich/crashs:latest`. Use the command below to download the container.

```sh
docker pull pyushkevich/crashs:latest
```

If you are using newer Mac with the ARM processor, you may need to use the `-platform` flag to download the container:

```sh
docker pull --platform linux/amd64 pyushkevich/crashs:latest
```

## Downloading CRASHS Templates and Models 

Before using CRASHS, you will need to download the templates and pretrained models from this link:

* https://doi.org/10.5061/dryad.kkwh70scx

Download and extract the archive and set the environment variable `CRASHS_DATA` to point to the folder in which you extract the archive.

```sh
cp ~/Downloads/crashs_template_package_20240830.tgz /my/crashs/folder
cd /my/crashs/folder
tar -zxvf crashs_template_package_20240830.tgz
export CRASHS_DATA=/my/crashs/folder/crashs_template_package_20240830
```

We recommend adding the line above that sets the `CRASHS_DATA` environment variable to your `.bashrc`, `.bash_profile` or `.zshrc` file, depending on what shell you use. Alternatively, you can invoke CRASHS below with the `-C` switch to provide the path to the templates and models directory.

## Inputs to CRASHS
The main input to the package is the ASHS output folder. Before running CRASHS, you will need to run ASHS on your MRI scans using one of the atlases for which a CRASHS template is available. 

CRASHS offers different templates for different ASHS versions. Currently, the following templates are provided:

* **ashs_pmc_t1**: Template for the T1-weighted MRI version of ASHS [T1-ASHS](https://doi.org/10.1002/hbm.24607) using the [ASHS-PMC-T1 atlas](https://www.nitrc.org/frs/?group_id=370). We recommend using the **2023 ASHS-PMC-T1 atlas with the white matter label**. However, you can also provide segmentations created using the original ASHS-PMC-T1 atlas and the white matter label will be added to the existing segmentation automatically, using [nnUNet](https://github.com/MIC-DKFZ/nnUNet). 

* **ashs_pmc_alveus**: Template for the high-resolution oblique coronal T2-weighted MRI version of [ASHS](https://doi.org/10.1002/hbm.22627). This template should be used with the **ASHS PMC** atlas. The white matter label will be added to the existing segmentation and extended synthetically over the alveus/fimbria, as described in our [ADNI 20th anniversary paper](https://doi.org/10.1002/alz.14161).

## Running CRASHS on a sample dataset

A sample dataset is included in the `sample_data` folder in the repository. Download it to some folder on your system (we will use `/my/crashs/folder/sample_data` for this tutorial). 

### Instructions for Docker

If using Docker, run the following command to open a command prompt on the container (change `/my/crashs/folder` to the right folder). 

```sh
docker run \
    -v your_output_directory:/data \
    -v /my/crashs/folder/crashs_template_package_20240830:/package \
    -v /my/crashs/folder/sample_data:/data \
    -it pyushkevich/crashs:latest /bin/bash
```

Run this command inside of the container to run CRASHS on the example T1-ASHS segmentation.

```sh
python3 -m crashs fit \
    -C /package -s right -c corr_usegray \
    /sample_data/ashs_pmc_t1/subj01/ashs ashs_pmc_t1 /sample_data/ashs_pmc_t1/subj01/crashs
```

You should find the output from running CRASHS in folder `/my/crashs/folder/sample_data/ashs_pmc_t1/subj01` on your system.

### Instructions for `pip` install

If using CRASHS installed with `pip` and the `CRASHS_DATA` environment variable has been set as explained above, use the command below to run CRASHS on the on the example T1-ASHS segmentation:

```sh
python3 -m crashs fit \
    -s right -c corr_usegray \
    /my/crashs/folder/sample_data/ashs_pmc_t1/subj01/ashs \
    ashs_pmc_t1 \
    /my/crashs/folder/sample_data/ashs_pmc_t1/subj01/crashs
```

You should find the output from running CRASHS in folder `/my/crashs/folder/sample_data/ashs_pmc_t1/subj01/crashs`.

### T2 Example

Another example in the `sample_data` folder can be used to test CRASHS for T2-weighted MRI processed with the ASHS-PMC atlas. It is better to run this example on a machine with an NVidia GPU because a nnU-Net is used by CRASHS to generate the white matter label; otherwise expect it to take 30-60 minutes to complete. If using Docker, include the flag `--gpus all` when calling the `docker run` command to make the GPU available to the container.

You can run the example in the Docker container like this:

```sh
python3 -m crashs fit -C /package -s left -c heur \
    /data/ashs_pmc_alveus/subj02/ashs \
    ashs_pmc_alveus \
    /data/ashs_pmc_alveus/subj02/crashs
```

Or using CRASHS pip install like this:

```sh
python3 -m crashs fit -s left -c heur \
    /my/crashs/folder/sample_data/ashs_pmc_alveus/subj02/ashs \
    ashs_pmc_alveus \
    /my/crashs/folder/sample_data/ashs_pmc_alveus/subj02/crashs
```

## Outputs from CRASHS
The program generates many outputs, but the most useful ones are:
* `fitting/[ID]_fitted_omt_hw_target.vtk`: the grey/white and grey/csf boundaries estimated by the `cruise_cortex_extraction` module of NighRes. These meshes are in physical (RAS) coordinate space, not in voxel (IJK) space output by Nighres. *If you extract meshes from the T1-ASHS segmentation in ITK-SNAP, those should line up with these meshes.*

* `fitting/[ID]_fitted_omt_hw_target.vtk`: the mid-surface of the gray matter estimated by the `volumetric_layering` module of NighRes. Also in RAS space.

* `fitting/[ID]_fitted_omt_match_to_hw.vtk`: the template mesh projected onto the mid-surface surface, also in RAS space. This should have the same geometry as the mid-surface, but the same number of vertices/faces as the template. This mesh will also have scalar arrays for the anatomical label and other features from the template, such as template curvature (useful for visualization). **This mesh can be used to map data from subject space (thickness, fMRI, NODDI, etc) into template space for group analysis**

* `thickness/[ID]_template_thickness.vtk`: a mesh with same geometry as the template that has a point array `VoronoiRadius` containing half-thickness of the gray matter at each vertex.

* `thickness/[ID]_thickness_roi_summary.csv`: Mean and median half-thickness across gray matter ROIs.

The following files can be used to check how well the fitting between the inflated template mid-surface and the inflated subject mid-surface worked.

* `fitting/[ID]_fit_target_reduced.vtk`: this is the inflated and sub-sampled mid-surface mesh of the subject, affine transformed into the space of the inflated template. Each triangle is associated with an anatomical label.

* `fitting/[ID]_fitted_lddmm_template.vtk`: this is the inflated template warped to optimally match the mesh above. The fit is not perfect but should be close.

* `fitting/[ID]_fitted_dist_stat.json`: distance statistics of the fitting, including average, max, and 95th percentile of the distance. Useful to check for poor fitting results.

## Citation
* PA Yushkevich, L Xie, LEM Wisse, M Dong, S Ravikumar, R Ittyerah,  R de Flores, SR Das, DA Wolk for the Alzheimer’s Disease Neuroimaging Initiative (ADNI), Mapping Medial Temporal Lobe Longitudinal Change in Preclinical Alzheimer’s Disease, 2023 Alzheimer's Association International Conference (AAIC 2023).


    

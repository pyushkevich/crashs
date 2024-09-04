# CRASHS: Cortical Reconstruction for Automatic Segmentation of Hippocampal Subfields (ASHS)
CRASHS is a surface-based modeling and groupwise registration pipeline for the human medial temporal lobe (MTL). It is used to postprocess the results of [ASHS](https://github.com/pyushkevich/ashs) segmentation with ASHS atlases that contain a white matter label. CRASHS uses the [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043) technique implemented in the [NighRes software](https://nighres.readthedocs.io/en/latest/) to fit the white matter segmentation with a surface of spherical topology, and find a series of surfaces spanning between the gray/white boundary and the pial surface. The middle surface is inflated and registered to a population template, allowing surface-based analysis of MTL cortical thickness and other measures such as functional MRI and diffusion MRI. 

The CRASHS pipeline is described in the supplemental material to our paper in the special issue of Alzheimer's and Dementia on the [20th anniversary of ADNI](https://doi.org/10.1002/alz.14161).


## Installation using `pip`

CRASHS requires the `nighres` package, which cannot be installed with `pip`. To install `nighres`, please follow the [installation instructions](https://nighres.readthedocs.io/en/latest/). To our knowledge, ARM64 architecture is currently not supported.

Once `nighres` is installed, you can install CRASHS:

```sh
pip install crashs
python3 -m crashs fit --help
```

Or, if you want to run the latest development code:

```sh
git clone https://github.com/pyushkevich/crashs
cd crashs
pip install .
```

## Docker
This repository includes the CRASHS scripts and a `Dockerfile`. The official container on DockerHub is labeled `pyushkevich/crashs:latest`

```sh
docker run -v your_data_directory:/data -it pyushkevich/crashs:latest /bin/bash
python3 -m crashs fit --help
```

A sample dataset is also provided and can be processed as follows (also see `run_sample.sh`)

```sh
docker run -v your_data_directory:/data -it pyushkevich/crashs:latest /bin/bash
python3 -m crashs fit -s right -r 0.1 sample_data/035_S_4082_2011-06-28 templates/crashs_template_01 /data/test
```

## Inputs to CRASHS
The main input to the package is the ASHS output folder. Before running CRASHS, you will need to run ASHS on your MRI scans using one of the atlases for which a CRASHS template is available. 

CRASHS offers different templates for different ASHS versions. Currently, the following templates are provided:

* **ashs_pmc_t1**: Template for the T1-weighted MRI version of ASHS [T1-ASHS](https://doi.org/10.1002/hbm.24607) using the [ASHS-PMC-T1 atlas](https://www.nitrc.org/frs/?group_id=370). We recommend using the 2023 ASHS-PMC-T1 atlas with the white matter label. However, you can also provide segmentations created using the original ASHS-PMC-T1 atlas and the white matter label will be added to the existing segmentation automatically, using [nnUNet](https://github.com/MIC-DKFZ/nnUNet). 

* **ashs_pmc_alveus**: Template for the high-resolution oblique coronal T2-weighted MRI version of [ASHS](https://doi.org/10.1002/hbm.22627). The white matter label will be added to the existing segmentation and extended synthetically over the alveus/fimbria, as described in our [ADNI 20th anniversary paper](https://doi.org/10.1002/alz.14161).

## Downloading CRASHS Templates and Models 

You can download the templates and pretrained models for running CRASHS from this link:

* https://doi.org/10.5061/dryad.kkwh70scx

Download and extract the archive and set the environment variable `CRASHS_DATA` to point to the folder in which you extract the archive.

```sh
cp ~/Downloads/crashs_template_package_20240830.tgz /my/crashs/folder
cd /my/crashs/folder
tar -zxvf crashs_template_package_20240830.tgz
export CRASHS_DATA=/my/crashs/folder/crashs_template_package_20240830
```

We recommend adding the line above that sets the `CRASHS_DATA` environment variable to your `.bashrc`, `.bash_profile` or `.zshrc` file, depending on what shell you use. Alternatively, you can invoke CRASHS below with the `-C` switch to provide the path to the templates and models directory.
 
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


    

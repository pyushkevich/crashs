# crashs=Cruise+ASHS
CRASHS is a surface-based modeling and groupwise registration pipeline for the human medial temporal lobe (MTL). It is used to postprocess the results of [ASHS](https://github.com/pyushkevich/ashs) segmentation with ASHS atlases that contain a white matter label. CRASHS uses the [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043) technique implemented in the [NighRes software](https://nighres.readthedocs.io/en/latest/) to fit the white matter segmentation with a surface of spherical topology, and find a series of surfaces spanning between the gray/white boundary and the pial surface. The middle surface is inflated and registered to a population template, allowing surface-based analysis of MTL cortical thickness and other measures such as functional MRI and diffusion MRI. 

## Inputs
The main input to the package is the ASHS-T1 output folder. ASHS should be run using the new [ASHS-T1 atlas with the white matter label](https://www.nitrc.org/frs/downloadlink.php/13554).

## Outputs
The program generates many outputs, but the most useful ones are:
* `fitting/[ID]_fitted_omt_hw_target.vtk`: the grey/white and grey/csf boundaries estimated by the `cruise_cortex_extraction` module of NighRes. These meshes are in physical (RAS) coordinate space, not in voxel (IJK) space output by Nighres. *If you extract meshes from the T1-ASHS segmentation in ITK-SNAP, those should like up with these meshes.*

* `fitting/[ID]_fitted_omt_hw_target.vtk`: the mid-surface of the gray matter estimated by the `volumetric_layering` module of NighRes. Also in RAS space.

* `fitting/[ID]_fitted_omt_match_to_hw.vtk`: the template mesh projected onto the mid-surface surface, also in RAS space. This should have the same geometry as the mid-surface, but the same number of vertices/faces as the template. This mesh will also have scalar arrays for the anatomical label and other features from the template, such as template curvature (useful for visualization). **This mesh can be used to map data from subject space (thickness, fMRI, NODDI, etc) into template space for group analysis**

The following files can be used to check how well the fitting between the inflated template mid-surface and the inflated subject mid-surface worked.

* `fitting/[ID]_fit_target_reduced`: this is the inflated and sub-sampled mid-surface mesh of the subject, affine transformed into the space of the inflated template. Each triangle is associated with an anatomical label.

* `fitting/[ID]_fitted_lddmm_template`: this is the inflated template warped to optimally match the mesh above. The fit is not perfect but should be close.

* `fitting/[ID]_fitted_dist_stat.json`: distance statistics of the fitting, including average, max, and 95th percentile of the distance. Useful to check for poor fitting results.


## Docker
This repository includes the CRASHS scripts and a `Dockerfile`. The official container on DockerHub is labeled `pyushkevich/crashs:latest`

    docker run -v your_data_directory:/data -it pyushkevich/crashs:latest /bin/bash
    ./crashs.py --help
    
A sample dataset is also provided and can be processed as follows (also see `run_sample.sh`)

    docker run -v your_data_directory:/data -it pyushkevich/crashs:latest /bin/bash
    ./crashs.py -s right -r 0.1 sample_data/035_S_4082_2011-06-28 templates/crashs_template_01 /data/test

## Citation
* PA Yushkevich, L Xie, LEM Wisse, M Dong, S Ravikumar, R Ittyerah,  R de Flores, SR Das, DA Wolk for the Alzheimer’s Disease Neuroimaging Initiative (ADNI), Mapping Medial Temporal Lobe Longitudinal Change in Preclinical Alzheimer’s Disease, 2023 Alzheimer's Association International Conference (AAIC 2023).


    

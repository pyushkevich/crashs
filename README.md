# CRASHS: Cortical Reconstruction for Automatic Segmentation of Hippocampal Subfields (ASHS)
CRASHS is a surface-based modeling and groupwise registration pipeline for the human medial temporal lobe (MTL). It can be used to perform groupwise analysis of pointwise measures in the MTL, such as cortical thickness, longitudinal volume change, functional MRI activation, microstructure, etc. It uses similar principles to whole-brain surface-based analysis pipelines like [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) and [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043), but restricted to the MTL region. CRASHS is used to postprocess the results of [ASHS](https://github.com/pyushkevich/ashs) segmentation with certain ASHS atlases.

![CRASHS overview figure](docs/source/_static/fig_crashs_overview_adni_paper.png "CRASHS overview figure")

Some of the newer ASHS atlases include the white matter label, which is used by CRASHS. For other ASHS atlases, CRASHS can paint in the white matter label using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). CRASHS uses the [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043) technique implemented in the [NighRes software](https://nighres.readthedocs.io/en/latest/) to fit the white matter segmentation with a surface of spherical topology, and find a series of surfaces spanning between the gray/white boundary and the pial surface. The middle surface is inflated and registered to a population template, allowing surface-based analysis of MTL cortical thickness and other measures such as functional MRI and diffusion MRI. 

The CRASHS pipeline is described in the supplemental material to our paper in the special issue of Alzheimer's and Dementia on the [20th anniversary of ADNI](https://doi.org/10.1002/alz.14161).

## Installation using `pip`

CRASHS requires the `nighres` package, which cannot be installed with `pip`. To install `nighres`, please follow the [installation instructions](https://nighres.readthedocs.io/en/latest/).

Once `nighres` is installed, you can install CRASHS:

```sh
pip install crashs
python3 -m crashs --help
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

After pulling the Docker container, use the command below to open a bash shell in the docker container, which will allow you to execute CRASHS commands in the container. The folder `/my/crashs/folder/sample_data` below stands for a folder on your file system that contains ASHS outputs that you wish to process. The folder `/my/crashs/folder/crashs_template_package` is a folder to which CRASHS will download its template package. The first time your run CRASHS, this should be an empty folder that you create. 

```sh
docker run \
    -v your_output_directory:/data \
    -v /my/crashs/folder/crashs_template_package:/package \
    -v /my/crashs/folder/sample_data:/data \
    -it pyushkevich/crashs:latest /bin/bash
```


## Downloading CRASHS Templates and Models 

Before using CRASHS, you will need to download the templates and pretrained models. The models are stored on HuggingFace at https://huggingface.co/datasets/pyushkevich/crashs_template_package, and can be downloaded to a folder on your filesystem (in the example below, `/my/crashs/folder/crashs_template_package`) using:

```sh
python crashs download /my/crashs/folder/crashs_template_package
```

If running inside of the Docker container, the command is:

```sh
python crashs download /package
```

The same command can be used in the future to update the template package to the latest version. It is convenienet to set the environment variable `CRASHS_DATA` to point to the folder where the package was downloaded: 

```sh
export CRASHS_DATA=/my/crashs/folder/crashs_template_package
```

We recommend adding the line above that sets the `CRASHS_DATA` environment variable to your `.bashrc`, `.bash_profile` or `.zshrc` file, depending on what shell you use. Alternatively, you can invoke CRASHS below with the `-C` switch to provide the path to the templates and models directory.

## Inputs to CRASHS
The main input to the package is the ASHS output folder. Before running CRASHS, you will need to run ASHS on your MRI scans using one of the atlases for which a CRASHS template is available. 

CRASHS offers different templates for different ASHS versions. Currently, the following templates are provided:

* **ashs_pmc_t1**: Template for the T1-weighted MRI version of ASHS [T1-ASHS](https://doi.org/10.1002/hbm.24607) using the [ASHS-PMC-T1 atlas](https://www.nitrc.org/frs/?group_id=370). We recommend using the **2023 ASHS-PMC-T1 atlas with the white matter label**. However, you can also provide segmentations created using the original ASHS-PMC-T1 atlas and the white matter label will be added to the existing segmentation automatically, using [nnUNet](https://github.com/MIC-DKFZ/nnUNet). 

* **ashs_pmc_t1exst**: Template for the T1-weighted MRI version of ASHS [T1-ASHS](https://doi.org/10.1002/hbm.24607) using the [ASHS-PMC-T1ext atlas](https://www.nitrc.org/frs/?group_id=370). This atlas extends the MTL cortical structures (ERC, BA35, BA36) more anteriorly and also includes the amygdala and white matter labels.  

* **ashs_pmc_alveus**: Template for the high-resolution oblique coronal T2-weighted MRI version of [ASHS](https://doi.org/10.1002/hbm.22627). This template should be used with the **ASHS PMC** atlas. The white matter label will be added to the existing segmentation and extended synthetically over the alveus/fimbria, as described in our [ADNI 20th anniversary paper](https://doi.org/10.1002/alz.14161).

## Running CRASHS on a sample dataset

A sample dataset is included in the `sample_data` folder in the repository. Download it to some folder on your system (we will use `/my/crashs/folder/sample_data` for this tutorial). 

### Instructions for Docker

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
* `thickness/[ID]_template_thickness.vtk`: a mesh with same number of vertices and faces as the CRASHS template that has been fitted to the mid-surface of the cortex and that contains point array `VoronoiRadius` that estimates half-thickness of the cortex at each vertex. This is the main output to use for downstream statistical analysis. Additionally, array `plab` contains the posterior probability of each anatomical label defined in the template. Value `0` corresponds to the white matter, and thickness values there should be ignored (most of them will be `NaN` anyway). **These meshes can be used for groupwise analysis of cortical thickness**.

* `thickness/[ID]_thickness_roi_summary.csv`: Mean and median of the `VoronoiRadius` array in `thickness/[ID]_template_thickness.vtk` integrated over ROIs defined in the template.

* `fitting/[ID]_fitted_omt_match_to_p00.vtk ... fitting/[ID]_fitted_omt_match_to_p09.vtk`: these meshes are similar to `thickness/[ID]_template_thickness.vtk` in that they represent the template's geometry fitted to the subject's cortex, but they are fitted to different layers: `00` corresponds to the gray-white surface and `09` to the pial surface. **These meshes can be used to sample data from the cortex in subject space (fMRI, NODDI, etc) into template space for group analysis**

The following files can be used to check how well the fitting between the inflated template mid-surface and the inflated subject mid-surface worked.

* `fitting/[ID]_fit_target_reduced.vtk`: this is the inflated and sub-sampled mid-surface mesh of the subject, affine transformed into the space of the inflated template. Each triangle is associated with an anatomical label.

* `fitting/[ID]_fitted_lddmm_template.vtk`: this is the inflated template warped to optimally match the mesh above. The fit is not perfect but should be close.

* `fitting/[ID]_fitted_dist_stat.json`: distance statistics of the fitting, including average, max, and 95th percentile of the distance. Useful to check for poor fitting results.

## CRASHS command-line parameters

Run `python3 -m crashs fit --help` to print the command-line parameters.

One set of parameters is used to specify which ASHS output should be used for fitting the geometrical representation:

* `-s {left,right}` is used to specify the side of the brain that should be fitted
* `-f {multiatlas,bootstrap}` is used to specify whether to use the ASHS output from the initial multi-atlas stage or the second bootstrap stage. Typically the bootstrap stage segmentation is better (accuracy is higher, on average, in ASHS validation experiments), so the default setting of `bootstrap` should be used.
* `-c {heur,corr_usegray,corr_nogray}` is used to specify which correction mode in ASHS should be used. The `heur` mode does not use any pixel-level machine learning correction and typically corresponds to smoothest shape segmentations. If the data on which you run ASHS is not well matched to the data on which ASHS was trained, it is best to use the `heur` option. The `corr_usegray` mode uses pixel-level machine learning correction, and in our validation experiments, has highest accuracy, but only if the data being segmented is similar to the training data (similar MRI protocol, age, etc.). Finally `corr_nogray` is an intermediate option that is rarely used.

The other parameters you may need to set are `-i` (specify the ID of the subject, used as a prefix in CRASHS output files), `-d` (specifies the device to use for PyTorch, e.g., `cuda0` if you have an NVidia GPU, `cpu` otherwise, and `-C` (to point to the templates and models folder if you didn't set the `CRASHS_DATA` environment variable). 

The options starting with `--skip` are used to skip certain steps when re-running CRASHS in the same folder. They are mostly used for debugging.

## Citations

* Yushkevich PA, Ittyerah R, Li Y, et al. Morphometry of medial temporal lobe subregions using high-resolution T2-weighted MRI in ADNI3: Why, how, and what's next? Alzheimer's Dement. 2024; 20: 8113–8128. https://doi.org/10.1002/alz.14161

* PA Yushkevich, L Xie, LEM Wisse, et al., Mapping Medial Temporal Lobe Longitudinal Change in Preclinical Alzheimer’s Disease, 2023 Alzheimer's Association International Conference (AAIC 2023).


    

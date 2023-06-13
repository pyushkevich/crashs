# crashs=Cruise+ASHS
CRASHS is a surface-based modeling and groupwise registration pipeline for the human medial temporal lobe (MTL). It is used to postprocess the results of [ASHS](https://github.com/pyushkevich/ashs) segmentation with ASHS atlases that contain a white matter label. CRASHS uses the [CRUISE](https://doi.org/10.1016/j.neuroimage.2004.06.043) technique implemented in the [NighRes software](https://nighres.readthedocs.io/en/latest/) to fit the white matter segmentation with a surface of spherical topology, and find a series of surfaces spanning between the gray/white boundary and the pial surface. The middle surface is inflated and registered to a population template, allowing surface-based analysis of MTL cortical thickness and other measures such as functional MRI and diffusion MRI. 

## Inputs
The main input to the package is the ASHS output folder.

## Docker
This repository includes the CRASHS scripts and a `Dockerfile`. The official container on DockerHub is labeled `pyushkevich/crashs:latest`

    docker run -v your_data_directory:/data -it pyushkevich/crashs:latest /bin/bash
    ./crashs.py --help
    
A sample dataset is also provided and can be processed as follows (also see `run_sample.sh`)

    docker run -v your_data_directory:/data -it pyushkevich/crashs:latest /bin/bash
    ./crashs.py -s right -r 0.1 sample_data/035_S_4082_2011-06-28 templates/crashs_template_01 /data/test

    

#!/bin/bash

# Download the atlas package 
python3 -m crashs download /package

# Run a sample dataset
python3 -m crashs fit \
    -C /package -s right -c corr_usegray \
    sample_data/ashs_pmc_t1/subj01/ashs \
    ashs_pmc_t1 \
    sample_data/ashs_pmc_t1/subj01/crashs

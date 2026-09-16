#!/bin/bash
SAMPLE_DATA_DIR=${1?}
TEMPLATE_PACKAGE_DIR=${2?}
OUTPUT_DIR=${3?}

for side in right; do
    python3 -m crashs fit \
        -S ${SAMPLE_DATA_DIR}/ashs_pmc_t1/subj01/ashs/bootstrap/fusion/posterior_corr_usegray_${side}_%03d.nii.gz \
        -A ${SAMPLE_DATA_DIR}/ashs_pmc_t1/subj01/ashs/affine_t1_to_template/t1_to_template_affine.mat \
        -i subj01 -s $side \
        -T ashs_pmc_t1 -C ${TEMPLATE_PACKAGE_DIR} \
        -w ${OUTPUT_DIR}/crashs_subj01_${side}
done
#!/bin/bash
SAMPLE_DATA_DIR=${1?}
TEMPLATE_PACKAGE_DIR=${2?}
OUTPUT_DIR=${3?}

for side in left right; do
    python3 -m crashs fit \
        -S ${SAMPLE_DATA_DIR}/ashs_pmc_alveus/subj02/ashs/final/subj02_${side}_lfseg_heur.nii.gz \
        -A ${SAMPLE_DATA_DIR}/ashs_pmc_alveus/subj02/ashs/affine_t1_to_template/t1_to_template_affine.mat \
        --tse-native-chunk ${SAMPLE_DATA_DIR}/ashs_pmc_alveus/subj02/ashs/tse_native_chunk_${side}.nii.gz \
        --mprage ${SAMPLE_DATA_DIR}/ashs_pmc_alveus/subj02/ashs/mprage.nii.gz \
        --affine-tse-to-mprage ${SAMPLE_DATA_DIR}/ashs_pmc_alveus/subj02/ashs/flirt_t2_to_t1/flirt_t2_to_t1.mat \
        -i subj02 -s $side \
        -T ashs_pmc_alveus -C ${TEMPLATE_PACKAGE_DIR} \
        -w ${OUTPUT_DIR}/crashs_subj02_${side}
    
    exit 255
done
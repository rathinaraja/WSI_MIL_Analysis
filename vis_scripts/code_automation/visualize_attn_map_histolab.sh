#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================

# # Fixed Paths
# WSI_DIR="/data_64T_3/Dataset/CAMELYON16/images/Tumor"
# FEATURE_DIR="/data_64T_3/Raja/MUFASA/VISUALIZATION/HISTOLAB/Extracted_features/resnet50_1024"
# COORDS_DIR="/data_64T_3/Raja/MUFASA/VISUALIZATION/HISTOLAB/Coordinates"
# BASE_OUTPUT_DIR="/data_64T_3/Raja/MUFASA/VISUALIZATION/ATTENTION_MAPS"

# # Model Settings
# MODEL_NAME="CLAM_MB_MIL"
# MODEL_CONFIG="configs/CLAM_MB_MIL.yaml"
# CHECKPOINT="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/logs/HISTOLAB/CAMELYON16/CLAM_MB_MIL/resnet50_1024/time_2026-01-05-20-46_CAMELYON16_CLAM_MB_MIL_seed_2023_resnet50_1024/fold_1/Best_EPOCH_63.pth"  
# WSI_LIST=("tumor_063" "tumor_071" "tumor_079" "tumor_086")

# Fixed Paths
WSI_DIR="/data_64T_3/Dataset/TCGA_LUSC/images/Tumor_DX1"
FEATURE_DIR="/data_64T_3/Raja/MUFASA/EXTRACTED_TILES_FEATURES_COORDINATES/HISTOLAB/Extracted_features/resnet50_1024"
COORDS_DIR="/data_64T_3/Raja/MUFASA/EXTRACTED_TILES_FEATURES_COORDINATES/HISTOLAB/Coordinates"
BASE_OUTPUT_DIR="/data_64T_3/Raja/MUFASA/VISUALIZATION/TCGA_NSCLC/ATTENTION_MAPS"

# Model Settings
MODEL_NAME="CLAM_MB_MIL"
MODEL_CONFIG="configs/CLAM_MB_MIL.yaml"
CHECKPOINT="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/logs/HISTOLAB/TCGA_NSCLC/CLAM_MB_MIL/resnet50_1024/time_2026-01-08-02-41_TCGA_NSCLC_CLAM_MB_MIL_seed_42_resnet50_1024/fold_2/Best_EPOCH_104.pth" 

# =================================================================
# DEFINE SLIDE LIST
# =================================================================

# Correct Syntax: No spaces around '=', use parentheses for array
WSI_LIST=("TCGA-52-7811-01Z-00-DX1.7093a626-0e8f-4e4c-80c6-6cf32a2f725e" "TCGA-52-7809-01Z-00-DX1.91f184ce-e4e1-42da-bdbc-36d6781bbf31" "TCGA-34-2604-01Z-00-DX1.C04E4FF6-6E62-432D-AD1E-D0AACAA66875" "TCGA-56-8307-01Z-00-DX1.20EE4B4C-403F-44AA-8B0E-711003E60B8F" "TCGA-60-2696-01Z-00-DX1.17748315-09b2-4abd-97f1-93c9951b0a70" "TCGA-77-8128-01Z-00-DX1.5831331a-8c82-4817-977e-1842250d9c7b" "TCGA-77-8136-01Z-00-DX1.15cdacc3-ee04-4323-b5e4-4f6d7085bd38" "TCGA-77-8138-01Z-00-DX1.fac6dcf9-7367-4345-a764-38e5471763c0" "TCGA-77-8139-01Z-00-DX1.242d68e3-5ee0-4484-8403-e2ffd1ede20b" "TCGA-77-8143-01Z-00-DX1.e844a7e1-ebba-4acb-8fe2-08aee2102848" "TCGA-77-8150-01Z-00-DX1.8ae93dac-2038-4426-9ddf-6fb857f6938a" "TCGA-77-A5G3-01Z-00-DX1.7ED6F76A-333F-43DB-B8DF-86BF259DCEFB" "TCGA-85-6175-01Z-00-DX1.102f9bad-4084-4e3a-99e9-91e026ad9a62" "TCGA-85-6798-01Z-00-DX1.c8ca3a56-b337-4345-b55a-83b0ae9a75f0" "TCGA-85-7696-01Z-00-DX1.d8756b4c-819f-4a5c-b148-125b8c6b3c27" "TCGA-85-8048-01Z-00-DX1.1a663caa-dff1-4f03-b06b-c8b8b5ce08c6" "TCGA-85-8070-01Z-00-DX1.54e7edf1-28d3-4fd5-bab3-951e58620386" "TCGA-94-7557-01Z-00-DX1.ac61f900-6f9f-44eb-91ad-98c98af8b741" "TCGA-NC-A5HM-01Z-00-DX1.CBAF06E0-5185-4A9C-B005-F7D2C0945032" "TCGA-NC-A5HP-01Z-00-DX1.655093A9-AAA7-4637-A33E-90FE3AE2FC43" "TCGA-86-8076-01Z-00-DX1.e7378b2f-e20e-4d2f-a86c-3a8ead08a385")

# =================================================================
# EXECUTION LOOP
# =================================================================

# Loop through the array defined above
# "${WSI_LIST[@]}" expands to the items in your list
for WSI_NAME in "${WSI_LIST[@]}"
do
    echo "----------------------------------------------------------------"
    echo "Processing Slide: $WSI_NAME"
    echo "----------------------------------------------------------------"

    # Define specific output folder for this slide
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${WSI_NAME}/HISTOLAB"

    python visualize_heatmaps_single_wsi.py \
        --wsi_dir "$WSI_DIR" \
        --feature_dir "$FEATURE_DIR" \
        --coords_dir "$COORDS_DIR" \
        --wsi_name "$WSI_NAME" \
        --model_name "$MODEL_NAME" \
        --model_config "$MODEL_CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR"

    echo "Done with $WSI_NAME."
    echo ""
done
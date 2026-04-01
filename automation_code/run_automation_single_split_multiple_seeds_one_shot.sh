#!/bin/bash

# ======================= CONDA INITIALIZATION =======================
# This block is required to allow 'conda activate' to work within a script
# It attempts to locate conda and source the setup script.
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    # Fallback: try to find it dynamically
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Error: Could not find conda.sh. Please ensure conda is in your PATH."
        exit 1
    fi
fi

# ======================= GLOBAL PARAMETERS =======================
PRE_PROCESSOR="MUFASA"  # CLAM, TRIDENT, HISTOLAB, MUFASA 
ROOT_PATH="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets"
NUM_EPOCHS=200
LOG_ROOT_DIR="logs/${PRE_PROCESSOR}" 

# Arrays for iteration
DATASETS=("COAD_READ")   # TCGA_STAD TCGA_NSCLC

# Parallel Arrays for Feature Extractors and their Dimensions
# Index 0 of EXTRACTORS corresponds to Index 0 of DIMS, etc.
EXTRACTORS=("resnet50_1024_set1")   
# EXTRACTORS=("resnet50_1024_set1" "resnet18")
# Make sure to provide resnet50_1024_set1/resnet50_1024_set1_set2_set3_combined and resnet18_set1/resnet18_set1_set2_set3_combined for mufasa

DIMS=(1024)   
# DIMS=(1024 512)

# Parallel Arrays for Seeds and Devices (Batch of 5)
SEEDS=(41 1337 2023 3407 9999)
DEVICES=(0 1 2 3 4)

# ======================= TRAINING FUNCTION =======================
# This function uses the Global parameters defined above.
# It iterates through the currently defined 'MODELS' array.
train_model_batch() {
    local CURRENT_ENV=$(conda info | grep "active environment" | cut -d : -f 2 | tr -d ' ')
    echo "================================================================================="
    echo "   RUNNING BATCH IN ENVIRONMENT: $CURRENT_ENV"
    echo "================================================================================="

    # 1. Loop through Datasets
    for DATASET in "${DATASETS[@]}"; do
        
        # 2. Loop through Feature Extractors
        for i in "${!EXTRACTORS[@]}"; do
            FEATURE_EXTRACTOR="${EXTRACTORS[$i]}"
            FEATURE_DIM="${DIMS[$i]}"
            
            # Construct the specific dataset directory path
            DATA_DIR="${ROOT_PATH}/${DATASET}/${DATASET}_${PRE_PROCESSOR}_${FEATURE_EXTRACTOR}_splits"

            # 3. Loop through Models (Uses the global MODELS array set before calling function)
            for MODEL_NAME in "${MODELS[@]}"; do
                
                echo "---------------------------------------------------------------------------------"
                echo "STARTING: ${MODEL_NAME} | ${DATASET} | ${FEATURE_EXTRACTOR} (Dim: ${FEATURE_DIM})"
                echo "Env: $CURRENT_ENV | Data: ${DATA_DIR}"
                echo "---------------------------------------------------------------------------------"

                # 4. Loop through the batch of Seeds/Devices (Launch 5 jobs in parallel)
                for j in "${!SEEDS[@]}"; do
                    SEED="${SEEDS[$j]}"
                    DEVICE="${DEVICES[$j]}"

                    python train_mil.py \
                    --yaml_path configs/${MODEL_NAME}.yaml \
                    --options Dataset.DATASET_NAME=${DATASET} \
                              Dataset.feature_extractor="${FEATURE_EXTRACTOR}" \
                              Model.in_dim=${FEATURE_DIM} \
                              Dataset.dataset_root_dir="${DATA_DIR}" \
                              Logs.log_root_dir="${LOG_ROOT_DIR}" \
                              General.seed=${SEED} \
                              General.num_epochs=${NUM_EPOCHS} \
                              General.device=${DEVICE} &
                done

                # Wait for the batch of 5 seeds to finish before starting the next model/config
                wait
                echo "Batch completed for ${MODEL_NAME}."
                echo ""
            done
        done
    done
}

# ======================= EXECUTION FLOW =======================

# --- PHASE 1: Standard MIL Models ---
echo "Activating: wsi_mil_classification"
conda activate wsi_mil_classification

# MODELS=(
#     "MEAN_MIL" "IB_MIL" "AB_MIL" "CLAM_MB_MIL" "DeepAttn_MIL" "DS_MIL" "TRANS_MIL"
#     "DTFD_MIL" "MHIM_MIL" "ILRA_MIL" "DGR_MIL" "MICO_MIL" "AC_MIL" "ADD_MIL" "TDA_MIL"
# )

MODELS=(
    "TDA_MIL"  
)

train_model_batch
#-------------------------------------

# # --- PHASE 2: Transformer Models ---
# echo "Activating: transformer"
# conda activate transformer

# MODELS=(
#      "PTC_MIL"  
# )

# train_model_batch
#-------------------------------------

# --- PHASE 3: Mamba MIL ---
echo "Activating: mambamil"
conda activate mambamil

MODELS=(
     "MAMBA_MIL"  
)

train_model_batch
#-------------------------------------

# --- PHASE 4: Mamba 2D MIL ---
echo "Activating: mambamil_2d"
conda activate mambamil_2d

MODELS=(
     "MAMBA2D_MIL" 
)

train_model_batch

echo "================================================================================="
echo "All experiments across all environments completed successfully."
echo "================================================================================="
#!/bin/bash

# ======================= CONDA INITIALIZATION =======================
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Error: Could not find conda.sh. Please ensure conda is in your PATH."
        exit 1
    fi
fi

# ======================= GLOBAL PARAMETERS =======================
PRE_PROCESSOR="CLAM"     # CLAM, TRIDENT, HISTOLAB, MUFASA 
ROOT_PATH="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets"
NUM_EPOCHS=200
LOG_ROOT_DIR="logs/${PRE_PROCESSOR}" 

DATASETS=("TCGA_NSCLC")  # TCGA_STAD TCGA_NSCLC

EXTRACTORS=("resnet50_1024")   
# EXTRACTORS=("resnet50_1024_set1" "resnet18")
# Make sure to provide resnet50_1024_set1/resnet50_1024_set1_set2_set3_combined and resnet18_set1/resnet18_set1_set2_set3_combined for mufasa

DIMS=(1024)   
# DIMS=(1024 512)

NUM_FOLDS=5
SEED=42
DEVICES=(0 1 2 3 4)

# ======================= TRAINING FUNCTION =======================
train_model_batch() {
    local CURRENT_ENV=$(conda info | grep "active environment" | cut -d : -f 2 | tr -d ' ')
    echo "================================================================================="
    echo "   RUNNING BATCH IN ENVIRONMENT: $CURRENT_ENV"
    echo "================================================================================="

    for DATASET in "${DATASETS[@]}"; do
        for i in "${!EXTRACTORS[@]}"; do
            FEATURE_EXTRACTOR="${EXTRACTORS[$i]}"
            FEATURE_DIM="${DIMS[$i]}"

            for MODEL_NAME in "${MODELS[@]}"; do
                
                echo "---------------------------------------------------------------------------------"
                echo "STARTING: ${MODEL_NAME} | ${DATASET} | ${FEATURE_EXTRACTOR} (Dim: ${FEATURE_DIM})"
                echo "Env: $CURRENT_ENV"
                echo "---------------------------------------------------------------------------------"

                # Loop 1 to 5 (folds)
                for ((j=1; j<=NUM_FOLDS; j++)); do

                    # Calculate correct device index (0-4)
                    DEVICE="${DEVICES[$j-1]}"

                    # 1. Define the folder where splits are located
                    SPLIT_DIR="${ROOT_PATH}/${DATASET}/${DATASET}_${PRE_PROCESSOR}_${FEATURE_EXTRACTOR}_splits"

                    # 2. Construct the specific CSV filename for this fold
                    # Example: Total_5-fold_TCGA_NSCLC_1fold_with_path.csv
                    CSV_FILE="Total_5-fold_${DATASET}_${j}fold_with_path.csv"
                    
                    # 3. Combine them to get the full path
                    FULL_CSV_PATH="${SPLIT_DIR}/${CSV_FILE}"
            
                    python train_mil.py \
                    --yaml_path configs/${MODEL_NAME}.yaml \
                    --options Dataset.DATASET_NAME=${DATASET} \
                              Dataset.feature_extractor="${FEATURE_EXTRACTOR}" \
                              Model.in_dim=${FEATURE_DIM} \
                              Dataset.dataset_csv_path="${FULL_CSV_PATH}" \
                              Logs.log_root_dir="${LOG_ROOT_DIR}" \
                              General.seed=${SEED} \
                              General.num_epochs=${NUM_EPOCHS} \
                              General.device=${DEVICE} &
                done

                # Wait for all 5 folds to finish
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

MODELS=(
    "MEAN_MIL" "IB_MIL" "AB_MIL" "CLAM_MB_MIL" "DeepAttn_MIL" "DS_MIL" "TRANS_MIL"
    "DTFD_MIL" "MHIM_MIL" "ILRA_MIL" "DGR_MIL" "MICO_MIL" "AC_MIL" "ADD_MIL" "TDA_MIL"
)

MODELS=(
    "MEAN_MIL"
    )

train_model_batch


# # --- PHASE 2: Transformer Models ---
# echo "Activating: transformer"
# conda activate transformer

# MODELS=(
#      "PTC_MIL"  
# )

# # train_model_batch


# # --- PHASE 3: Mamba MIL ---
# echo "Activating: mambamil"
# conda activate mambamil

# MODELS=(
#      "MAMBA_MIL"  
# )

# train_model_batch


# # --- PHASE 4: Mamba 2D MIL ---
# echo "Activating: mambamil_2d"
# conda activate mambamil_2d

# MODELS=(
#      "MAMBA2D_MIL" 
# )

# train_model_batch

echo "================================================================================="
echo "All experiments across all environments completed successfully."
echo "================================================================================="
#!/bin/bash

# echo "Starting MEAN_MIL..."

# --- MEAN_MIL --- 
python train_mil.py \
--yaml_path configs/MEAN_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/MEAN_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/MEAN_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/MEAN_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 & 

# echo "MEAN_MIL complete. Starting IB_MIL..."

# --- IB_MIL ---
python train_mil.py \
--yaml_path configs/IB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/IB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/IB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/IB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 & 

wait

# echo "IB_MIL complete. Starting AB_MIL..."

# --- AB_MIL --- 
python train_mil.py \
--yaml_path configs/AB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/AB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/AB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/AB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 &  

# echo "AB_MIL complete. Starting DeepAttn_MIL..."

# --- DeepAttn_MIL ---
python train_mil.py \
--yaml_path configs/DeepAttn_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/DeepAttn_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/DeepAttn_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/DeepAttn_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 &  

wait

# echo "DeepAttn_MIL complete. Starting CLAM_MB_MIL..."

# --- CLAM_MB_MIL ---
python train_mil.py \
--yaml_path configs/CLAM_MB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/CLAM_MB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/CLAM_MB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/CLAM_MB_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 &   

# echo "CLAM_MB_MIL complete. Starting DS_MIL..."

# --- DS_MIL ---
python train_mil.py \
--yaml_path configs/DS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/DS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/DS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/DS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 &  

wait

# echo "DS_MIL complete. Starting TRANS_MIL..."

# --- TRANS_MIL --- 
python train_mil.py \
--yaml_path configs/TRANS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/TRANS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/TRANS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/TRANS_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 &  

# echo "TRANS_MIL complete. Starting DTFD_MIL..."

# ---  DTFD_MIL ---
python train_mil.py \
--yaml_path configs/DTFD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/DTFD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/DTFD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/DTFD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 &  

wait

# echo "DTFD_MIL complete. Starting MHIM_MIL..."

# --- MHIM_MIL --- 
python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 & 
 
# echo "MHIM_MIL complete. Starting ILRA_MIL..."

# --- ILRA_MIL ---
python train_mil.py \
--yaml_path configs/ILRA_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/ILRA_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/ILRA_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/ILRA_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 & 

wait

# echo "ILRA_MIL complete. Starting DGR_MIL..."

# --- DGR_MIL--- 
python train_mil.py \
--yaml_path configs/DGR_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/DGR_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/DGR_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/DGR_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 & 

# echo "DGR_MIL complete. Starting MICO_MIL..."

# ---  MICO_MIL ---
python train_mil.py \
--yaml_path configs/MICO_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/MICO_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/MICO_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/MICO_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 & 

wait

# echo "MICO_MIL complete. Starting AC_MIL..."

# --- AC_MIL--- 
python train_mil.py \
--yaml_path configs/AC_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

python train_mil.py \
--yaml_path configs/AC_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

python train_mil.py \
--yaml_path configs/AC_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 &

python train_mil.py \
--yaml_path configs/AC_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 & 

# echo "AC_MIL complete. Starting ADD_MIL..."

# ---  ADD_MIL ---
python train_mil.py \
--yaml_path configs/ADD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/ADD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/ADD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/ADD_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_COAD_READ Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/TCGA_COAD_READ_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 & 

wait

echo "All jobs completed."
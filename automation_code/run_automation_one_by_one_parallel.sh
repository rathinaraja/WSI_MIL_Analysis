#!/bin/bash
# python train_mil.py \
# --yaml_path configs/MEAN_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/MEAN_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/MEAN_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/MEAN_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/IB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/IB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/IB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=6 &

# wait

# python train_mil.py \
# --yaml_path configs/IB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/AB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/AB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/AB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/AB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/DeepAttn_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/DeepAttn_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=6 &

# wait

# python train_mil.py \
# --yaml_path configs/DeepAttn_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/DeepAttn_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=1 & 

# python train_mil.py \
# --yaml_path configs/CLAM_MB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/CLAM_MB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/CLAM_MB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/CLAM_MB_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/DS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=6 &

# wait

# python train_mil.py \
# --yaml_path configs/DS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/DS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/DS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=2 & 

# python train_mil.py \
# --yaml_path configs/TRANS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/TRANS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/TRANS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/TRANS_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=6 &

# wait

# python train_mil.py \
# --yaml_path configs/DTFD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/DTFD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/DTFD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/DTFD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=3 & 

# python train_mil.py \
# --yaml_path configs/MHIM_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/MHIM_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/MHIM_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=6 &

# wait

# python train_mil.py \
# --yaml_path configs/MHIM_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/ILRA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/ILRA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/ILRA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/ILRA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=4 & 

# python train_mil.py \
# --yaml_path configs/DGR_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/DGR_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/DGR_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/DGR_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/MICO_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/MICO_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/MICO_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=6 &

# python train_mil.py \
# --yaml_path configs/MICO_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=7 & 

# wait

# python train_mil.py \
# --yaml_path configs/AC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=0 & 

# python train_mil.py \
# --yaml_path configs/AC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/AC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/AC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=3 &

# python train_mil.py \
# --yaml_path configs/ADD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

# python train_mil.py \
# --yaml_path configs/ADD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/ADD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=6 &

# python train_mil.py \
# --yaml_path configs/ADD_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=7 &

# wait

# python train_mil.py \
# --yaml_path configs/TDA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=0 &

# python train_mil.py \
# --yaml_path configs/TDA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/TDA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/TDA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=3 & 

# python train_mil.py \
# --yaml_path configs/MAMBA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=0 & 

# python train_mil.py \
# --yaml_path configs/MAMBA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &

# python train_mil.py \
# --yaml_path configs/MAMBA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=2 &

# python train_mil.py \
# --yaml_path configs/MAMBA_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=3 &

python train_mil.py \
--yaml_path configs/MAMBA2D_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=4 &

python train_mil.py \
--yaml_path configs/MAMBA2D_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &

python train_mil.py \
--yaml_path configs/MAMBA2D_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=6 &

python train_mil.py \
--yaml_path configs/MAMBA2D_MIL.yaml \
--options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=7 &

# python train_mil.py \
# --yaml_path configs/PTC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/CLAM' General.seed=42 General.num_epochs=200 General.device=4 & 

# python train_mil.py \
# --yaml_path configs/PTC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &

# python train_mil.py \
# --yaml_path configs/PTC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=6 &

# python train_mil.py \
# --yaml_path configs/PTC_MIL.yaml \
# --options Dataset.DATASET_NAME=TCGA_NSCLC Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/datasets/TCGA_NSCLC/TCGA_NSCLC_MUFASA_resnet50_1024_set1_splits' Logs.log_root_dir='logs/MUFASA' General.seed=42 General.num_epochs=200 General.device=7 &

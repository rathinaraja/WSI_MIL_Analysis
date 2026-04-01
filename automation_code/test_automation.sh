python train_mil.py \
--yaml_path configs/MEAN_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=0 &  

python train_mil.py \
--yaml_path configs/IB_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 &  

python train_mil.py \
--yaml_path configs/AB_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=2 &  

python train_mil.py \
--yaml_path configs/DeepAttn_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 &   

python train_mil.py \
--yaml_path configs/CLAM_MB_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=4 &    

python train_mil.py \
--yaml_path configs/DS_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &   

python train_mil.py \
--yaml_path configs/TRANS_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=6 &   

python train_mil.py \
--yaml_path configs/DTFD_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=7 &  

wait

python train_mil.py \
--yaml_path configs/ILRA_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=0 & 

python train_mil.py \
--yaml_path configs/DGR_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=1 & 

python train_mil.py \
--yaml_path configs/MICO_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=2 & 

python train_mil.py \
--yaml_path configs/AC_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=3 & 

python train_mil.py \
--yaml_path configs/ADD_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=4 & 

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_TRIDENT_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/TRIDENT' General.seed=42 General.num_epochs=200 General.device=5 &  

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_CLAM_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/CLAM' General.seed=42 General.num_epochs=200 General.device=6 &  

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_HISTOLAB_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/HISTOLAB' General.seed=42 General.num_epochs=200 General.device=7 &  

wait 

python train_mil.py \
--yaml_path configs/MHIM_MIL.yaml \
--options Dataset.DATASET_NAME=STANFORD_793 Dataset.feature_extractor="resnet50_1024" Model.in_dim=1024 Dataset.dataset_root_dir='datasets/MSS_MSI/STANFORD_793_MUFASA_resnet50_1024_splits' Logs.log_root_dir='logs/MSS_MSI/MUFASA' General.seed=42 General.num_epochs=200 General.device=0 &  


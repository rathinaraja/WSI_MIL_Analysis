import os
import argparse
import shutil
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.model_utils import get_model_from_yaml, get_criterion
from utils.yaml_utils import read_yaml
from utils.loop_utils import val_loop
from utils.wsi_utils import WSI_Dataset_Test, CDP_MIL_WSI_Dataset, LONG_MIL_WSI_Dataset

import warnings
warnings.filterwarnings('ignore')

def test(args):
    yaml_path = args.yaml_path
    print(f"MIL-model-yaml path: {yaml_path}")
    
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes
    
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    
    # CDP_MIL and LONG_MIL models have different dataset pipeline
    if model_name == 'CDP_MIL':
        test_ds = CDP_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.BeyesGuassian_pt_dir,'test')
        
    if model_name == 'LONG_MIL':
        test_ds = LONG_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
    else:
        test_ds = WSI_Dataset_Test(test_dataset_csv, args)
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    model_weight_path = args.model_weight_path
    print(f"Model weight path: {model_weight_path}")
    
    device = torch.device(f'cuda:{yaml_args.General.device}')
    criterion = get_criterion(yaml_args.Model.criterion)
    
    mil_model = get_model_from_yaml(yaml_args)
    mil_model = mil_model.to(device)
    mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))

    # UPDATED: Added inference_mode=True and threshold
    test_results = inference_loop(
        device, num_classes, mil_model, test_dataloader, criterion,
        inference_mode=True,  # ← ADDED
        threshold=0.5  # ← ADDED (use optimal threshold from training if available)
    )
    
    # UPDATED: Extract results
    predictions = test_results['predictions']
    probabilities = test_results['probabilities']
    wsi_ids = test_results['WSI_ids']
    
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print('----------------INFO----------------\n')
    print(f'{FAIL}Total Samples:{ENDC} {len(predictions)}\n')
    print(f'{FAIL}Predicted Positive:{ENDC} {sum(predictions)}\n')
    print(f'{FAIL}Predicted Negative:{ENDC} {len(predictions) - sum(predictions)}\n')
    
    test_log_dir = args.test_log_dir
    os.makedirs(test_log_dir, exist_ok=True)
    
    new_yaml_path = os.path.join(test_log_dir, f'Test_{model_name}.yaml')
    shutil.copyfile(yaml_path, new_yaml_path)
    
    new_test_dataset_csv_path = os.path.join(test_log_dir, f'Test_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
    shutil.copyfile(test_dataset_csv, new_test_dataset_csv_path)
    
    # UPDATED: Save predictions to CSV
    predictions_csv_path = os.path.join(test_log_dir, f'Predictions_{model_name}.csv')
    predictions_df = pd.DataFrame({
        'WSI_ID': wsi_ids,
        'Prediction': predictions,
        'Predicted_Class': ['Negative' if p == 0 else 'Positive' for p in predictions],
        'Prob_Class_0': probabilities[:, 0],
        'Prob_Class_1': probabilities[:, 1],
        'Confidence': [max(prob) for prob in probabilities]
    })
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to: {predictions_csv_path}")
    
    # UPDATED: Save summary log
    test_log_path = os.path.join(test_log_dir, f'Test_Log_{model_name}.txt')
    log_to_save = {
        'total_samples': len(predictions),
        'predicted_positive': int(sum(predictions)),
        'predicted_negative': int(len(predictions) - sum(predictions)),
        'average_confidence': float(np.mean([max(prob) for prob in probabilities])),
        'predictions': predictions.tolist(),
        'wsi_ids': wsi_ids
    }
    
    with open(test_log_path, 'w') as f:
        f.write(str(log_to_save))
    print(f"Test log saved at: {test_log_path}")
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='/path/to/your/config-yaml', help='path to MIL-model-yaml file')
    parser.add_argument('--test_dataset_csv', type=str, default='/path/to/your/ds-csv-path', help='path to dataset csv')
    parser.add_argument('--model_weight_path', type=str, default='/path/to/your/model-weight', help='path to model weights')
    parser.add_argument('--test_log_dir', type=str, default='/path/to/your/test-log-dir', help='path to test log dir')
    args = parser.parse_args()
    test(args)
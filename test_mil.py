import os
import argparse
import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.model_utils import get_model_from_yaml, get_criterion
from utils.yaml_utils import read_yaml
from utils.loop_utils import val_loop, clam_val_loop, ds_val_loop, dtfd_val_loop, inference_loop
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
    
    # Create dataset
    test_ds = WSI_Dataset_Test(test_dataset_csv, yaml_args)
    
    # Check if inference mode
    inference_mode = not test_ds.is_with_labels()
    if inference_mode:
        print("⚠️  Inference mode: No labels found in CSV. Only predictions will be generated.")
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    model_weight_path = args.model_weight_path
    print(f"Model weight path: {model_weight_path}")
    device = torch.device(f'cuda:{yaml_args.General.device}')
    
    # Load model
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier, attention, dimReduction, attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(model_weight_path, weights_only=True)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier, attention, dimReduction, attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))
        mil_model.eval()

    test_log_dir = args.test_log_dir
    os.makedirs(test_log_dir, exist_ok=True)
    
    if inference_mode:
        # Inference mode: only predictions
        predictions = []
        probabilities = []
        slide_names = []
        
        print("Running inference...")
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                bag = data[0].to(device).float()
                
                if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
                    # DTFD_MIL inference
                    classifier, attention, dimReduction, attCls = model_list
                    feat_index = torch.LongTensor([0]).to(device)
                    slide_pseudo_feat, _ = dimReduction(bag, feat_index)
                    slide_sub_preds = []
                    for subtyping_idx in range(yaml_args.Model.num_Group):
                        slide_sub_preds.append(attCls[subtyping_idx](slide_pseudo_feat))
                    logits = torch.stack(slide_sub_preds).mean(0)
                elif yaml_args.General.MODEL_NAME in ['CLAM_MB_MIL', 'CLAM_SB_MIL']:
                    # CLAM models return a dict
                    output = mil_model(bag)
                    logits = output['logits'] if isinstance(output, dict) else output
                elif yaml_args.General.MODEL_NAME == 'DS_MIL':
                    # DS_MIL returns (max_prediction, avg_prediction, max_instances)
                    max_pred, _, _ = mil_model(bag)
                    logits = max_pred
                else:
                    # Standard models
                    logits = mil_model(bag)
                
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().item()
                predictions.append(pred)
                probabilities.append(probs.cpu().numpy()[0].tolist())
                slide_names.append(os.path.basename(test_ds.slide_path_list[i]))
        
        # Save predictions
        results_df = pd.DataFrame({
            'slide_path': test_ds.slide_path_list,
            'slide_name': slide_names,
            'prediction': predictions,
            'probabilities': probabilities
        })
        
        predictions_path = os.path.join(test_log_dir, f'Predictions_{model_name}.csv')
        results_df.to_csv(predictions_path, index=False)
        
        print(f"\n✓ Predictions saved to: {predictions_path}")
        print(f"Total slides: {len(predictions)}")
        print(f"Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
        
    else:
        # Test mode with labels: compute metrics
        criterion = get_criterion(yaml_args.Model.criterion)
        
        if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL' or yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
            bag_weight = yaml_args.Model.bag_weight
            test_loss, test_metrics = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight)
        elif yaml_args.General.MODEL_NAME == 'DS_MIL':
            test_loss, test_metrics = ds_val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
            test_loss, test_metrics = dtfd_val_loop(device, num_classes, model_list, test_dataloader, criterion, 
                                                     yaml_args.Model.num_Group, yaml_args.Model.grad_clipping, 
                                                     yaml_args.Model.distill, yaml_args.Model.total_instance)
        else:
            test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}Test_Loss:{ENDC}{test_loss}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        
        new_yaml_path = os.path.join(test_log_dir, f'Test_{model_name}.yaml')
        shutil.copyfile(yaml_path, new_yaml_path)
        new_test_dataset_csv_path = os.path.join(test_log_dir, f'Test_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
        shutil.copyfile(test_dataset_csv, new_test_dataset_csv_path)
        test_log_path = os.path.join(test_log_dir, f'Test_Log_{model_name}.txt')
        log_to_save = {'test_loss': test_loss, 'test_metrics': test_metrics}
        with open(test_log_path, 'w') as f:
            f.write(str(log_to_save))
        print(f"Test log saved at: {test_log_path}")
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/config-yaml',help='path to MIL-model-yaml file')
    parser.add_argument('--test_dataset_csv',type=str,default='/path/to/your/ds-csv-path',help='path to dataset csv')
    parser.add_argument('--model_weight_path',type=str,default='/path/to/your/model-weight',help='path to model weights ')
    parser.add_argument('--test_log_dir',type=str,default='/path/to/your/test-log-dir',help='path to test log dir')
    args = parser.parse_args()
    test(args)


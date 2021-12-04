# ****************************************************************************
#  evaluate.py
# ****************************************************************************
#
#  Author:          Horst Osberger
#  Description:     Evaluation script to evaluate a trained network on a given
#                   dataset.
#                   The parameters can either be given by a config-file or
#                   separately.
#
#  (c) 2021 by Horst Osberger
# ****************************************************************************

import os
import sys
import argparse
import json

import torch
from torch.utils.data import DataLoader

# Add parent folder to include paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from source.functions import CustomImageDataset, Evaluater
from source.functions import initialize_model, get_device
from source.functions import get_base_folder, make_abs_path


def main():
    """
    Main function of this program
    """
    # Add a parser to handle optional parameters
    parser = argparse.ArgumentParser(\
        description='Evaluate a classifier on a whole dataset. ')
    parser.add_argument('--config_file', required=False, \
                        help='Path to the json config file with parameters. '
                             'If given, the keys \'model\', \'work_dir\', ' 
                             '\'test_dataset\', and \'val_top_k\' are selected '
                             'to identify which model, which dataset, and '
                             'which k should be evaluated. '
                             'If not given, all other params must be given. ')
    parser.add_argument('--model_name', required=False, \
                        help='Name of the model. '
                             'If not given, \'--config_file\' must be given.')
    parser.add_argument('--model_state_dict', required=False, \
                        help='Path to the model state_dict. '
                             'If not given, \'--config_file\' must be given.')
    parser.add_argument('--dataset', required=False, \
                        help='Path to the dataset used for evaluation. '
                             'If not given, \'--config_file\' must be given.')
    parser.add_argument('--top_k', required=False, \
                        help='k for top-k error. '
                             'If not given, \'--config_file\' must be given.')
    parser.add_argument('--use_cuda', action='store_true', \
                        help='Optional: If set, evaluation runs on GPU. ')
    parser.add_argument('--batch_size', type=int, default=1, \
                        help='Optional: Used batch size. Default = 1. ')
    parser.add_argument('--result_json', required=False, type=str, \
                        help='Optional: Results is saved to this json. '
                             'If \'--config_file\' is given, result is saved '
                             'to \'work_dir\'/evaluation.json. ')

    # Access parser arguments
    args = parser.parse_args()

    # Check inputs
    config_file = args.config_file
    model_name = args.model_name
    model_state_dict = args.model_state_dict
    dataset = args.dataset
    top_k = args.top_k
    batch_size = args.batch_size
    use_cuda = args.use_cuda

    base_folder = get_base_folder(config_file)

    config_set = config_file is not None
    others_set = (model_name is not None and model_state_dict is not None and\
                  dataset is not None and top_k is not None)

    if (not config_set and not others_set):
        raise Exception('No parameters set. Check --help for more information.')
    
    if (config_set and others_set):
        raise Exception('Either \'--config_file\' or all other parameters must '
                        'be set, but no a combination.')

    if config_set:
        # Read in the config file
        with open(config_file) as json_file:
            config = json.load(json_file)

        model_name = config['model']
        model_state_dict = os.path.join(config['work_dir'], \
                                        'best_model_state_dict.pt')
        model_state_dict = make_abs_path(base_folder, model_state_dict)
        dataset = config['val_dataset']
        top_k = config['val_top_k']

        # Save results to...
        result_json = os.path.join(config['work_dir'], 'evaluation.json')
        result_json = make_abs_path(base_folder, result_json)

    # Get dataloader
    dataset_path = make_abs_path(base_folder, dataset)
    dataset = CustomImageDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, \
                            shuffle=False, num_workers=0)

    # Initialize model
    class_ids, _ = dataloader.dataset.get_classes()
    model, _ = initialize_model(model_name, len(class_ids), \
                                use_pretrained = False)

    # Load weigths
    model.load_state_dict(torch.load(model_state_dict))
    model.eval()

    # Get device
    device = get_device(cuda = use_cuda)

    # Create and apply Evaluater
    print('Running evaluation on {} ... '.format(device.type))
    evaluater = Evaluater(dataloader, k = top_k, percentage = 1.0)
    accuracy, top_1_error, top_k_error = evaluater.eval(model, device)

    # Print results
    print('Evaluation results: ')
    print('\t accuracy:\t{}'.format(accuracy))
    print('\t top-1 error:\t{}'.format(top_1_error))
    print('\t top-{}-error:\t{}'.format(top_k, top_k_error))

    # Save results to json
    if result_json is not None:
        results = dict({'model': model_name, 'dataset': dataset_path, \
                        'accuracy': accuracy, \
                        'top_1_error': top_1_error, \
                        'top_{}_error'.format(top_k): top_k_error})
        with open(result_json, 'w') as fp:
            json.dump(results, fp, indent=4)
        print('Results saved to: {}'.format(result_json))

    return


if __name__ == '__main__':
    main()
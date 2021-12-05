# ****************************************************************************
#  predict_image.py
# ****************************************************************************
#
#  Author:          Horst Osberger
#  Description:     Predict a single image on a trained model.
#                   The parameters can either be given by a config-file or
#                   separately.
#
#  (c) 2021 by Horst Osberger
# ****************************************************************************

import os
import sys
import argparse
import time
import json

import torch
from torchvision.io import read_image

# Add parent folder to include paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from source.functions import CustomImageDataset
from source.functions import initialize_model, get_device, transform_image
from source.functions import get_num_classes_from_model_state
from source.functions import get_base_folder, make_abs_path


def main():
    """
    Main function of this program
    """
    # Add a parser to handle optional parameters
    parser = argparse.ArgumentParser(\
        description='Apply a classifier to a single image. ')
    parser.add_argument('image_path', type=str, \
                        help='Path to the image to be classified.')
    parser.add_argument('--config_file', required=False, \
                        help='Path to the json config file with parameters. '
                             'If given, the keys \'model\' and \'work_dir\', ' 
                             'are selected to identify the model to be used. '
                             'In addition, a dataset is used to retrieve the '
                             'class names. ')
    parser.add_argument('--model_name', required=False, \
                        help='Name of the model. '
                             'If not given, \'--config_file\' must be given.')
    parser.add_argument('--model_state_dict', required=False, \
                        help='Path to the model state_dict. '
                             'If not given, \'--config_file\' must be given.')
    parser.add_argument('--dataset', required=False, \
                        help='Optional: Use dataset to get class names. '
                             'Is ignored if \'--config_file\' is given.')
    parser.add_argument('--use_cuda', action='store_true', \
                        help='Optional: If set, evaluation runs on GPU. ')

    # Access parser arguments
    args = parser.parse_args()

    # Check inputs
    image_path = args.image_path
    config_file = args.config_file
    model_name = args.model_name
    model_state_dict = args.model_state_dict
    use_cuda = args.use_cuda
    dataset = args.dataset

    config_set = config_file is not None
    others_set = (model_name is not None and model_state_dict)

    if (not config_set and not others_set):
        raise Exception('Parameters missing. Check --help for more information.')
    
    if (config_set and others_set):
        raise Exception('Either \'--config_file\' or all other parameters must '
                        'be set, but no a combination.')

    base_folder = get_base_folder(config_file)
    if config_set:
        # Read in the config file
        with open(config_file) as json_file:
            config = json.load(json_file)

        model_name = config['model']
        model_state_dict = os.path.join(config['work_dir'], \
                                        'best_model_state_dict.pt')
        dataset = config['val_dataset']
    
    class_names = None
    if (dataset is not None):
        if (not os.path.isabs(dataset)):
          dataset = make_abs_path(base_folder, dataset)
        dataset = CustomImageDataset(dataset)
        class_ids, class_names = dataset.get_classes()

    # Initialize model and load weights
    model_state_dict = make_abs_path(base_folder, model_state_dict)
    state_dict = torch.load(model_state_dict)
    num_classes = get_num_classes_from_model_state(state_dict)
    model, input_size = initialize_model(model_name, num_classes, \
                                         use_pretrained = False)
    model.load_state_dict(state_dict)
    model.eval()

    # Read in model and preprocess
    img = read_image(image_path)
    img = transform_image(width=input_size, height=input_size)(img.unsqueeze(0))

    # Get device
    device = get_device(cuda = use_cuda)

    # Shift to device and predict
    model = model.to(device)
    img = img.to(device)
    start = time.time()
    output = model(img)
    _, preds = torch.max(output, 1)
    end = time.time()

    # Print results
    print('Prediction: ')
    print('\t class index:\t\t{}'.format(preds[0]))
    if (class_names is not None):
        print('\t class id:\t\t{}'.format(class_ids[preds[0]]))
        print('\t class name:\t\t{}'.format(class_names[preds[0]]))
    print('\t execution time:\t{:.2f} ms'.format(1000*(end - start)))

    return


if __name__ == '__main__':
    main()
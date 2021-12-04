# ****************************************************************************
#  train.py
# ****************************************************************************
#
#  Author:          Horst Osberger
#  Description:     Training script to run a single training on a given
#                   dataset and model.
#                   The parameters can either be given by a config-file or
#                   separately.
#
#  (c) 2021 by Horst Osberger
# ****************************************************************************

import os
import sys
import argparse
import matplotlib.pyplot as plt
import json
import time

# Add parent folder to include paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from source.functions import CustomImageDataset, Evaluater
from source.functions import initialize_model, train_model, get_device, \
                             calculate_loss_weights, plot_loss, plot_eval
from source.functions import transform_image, augment_image
from source.functions import get_base_folder, make_abs_path


def create_folder(directory):
    """
    Create a directory.

    Args:
        directory: Path to directory that should be created
    """
    try:
        os.mkdir(directory)
        return
    except:
        pass
    # Creation failed --> try to create base folder
    directory_split = os.path.split(directory)
    create_folder(directory_split[0])

    # Try again  
    try:
        os.mkdir(directory)
    except:
        pass
    return


def create_work_dir(work_dir):
    """
    Helper for main, checks and creates the working directory.

    Args:
        work_dir: Path to work_dir that should be created
    """
    if (os.path.isdir(work_dir)):
        raise ValueError('Working directory {} already exists. Delete folder '
                         'or rename work_dir in config file'.format(work_dir))

    create_folder(work_dir)
    if (not os.path.isdir(work_dir)):
        raise ValueError('Working directory {} could not be created'.format(work_dir))


def create_data_loaders(train_dataset, val_dataset, batch_size):
    """
    Create the data loaders for training and validation.

    Args:
        train_dataset: Path to training dataset
        val_dataset: Path to validation dataset
        batch_size: Used batch size

    Returns:
        Training and validation dataloaders
    """

    # Create dataset and loader for training
    train_dataset = CustomImageDataset(train_dataset, \
                            transform = transform_image(), \
                            augment = augment_image())
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
                                shuffle=True, num_workers=0)

    # Create evaluater class
    val_dataset = CustomImageDataset(val_dataset, \
                            transform = transform_image())
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, \
                                shuffle=True, num_workers=0)

    return train_dataloader, val_dataloader


def get_config(config_file):
    """
    Get a (adapted) config dict from the config json file.
    One adaption here is, that relative file paths are changed to absolute
    paths.

    Args:
        config_file: Path to the json config file

    Returns:
        Adapted config file
    """
    # Read in the config file
    with open(config_file) as json_file:
        config = json.load(json_file)

    # Change paths to be absolute...
    base_folder = get_base_folder(config_file)
    config['train_dataset'] = make_abs_path(base_folder, config['train_dataset'])
    config['val_dataset'] = make_abs_path(base_folder, config['val_dataset'])
    config['test_dataset'] = make_abs_path(base_folder, config['test_dataset'])
    config['work_dir'] = make_abs_path(base_folder, config['work_dir'])

    return config


def main():
    """
    Main function of this program
    """
    # Add a parser to handle optional parameters
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--config_file', required=True, \
                        help='Path to the json config file with parameters.')

    # Access parser arguments
    args = parser.parse_args()

    # Read in (and adapt) the config file
    config = get_config(args.config_file)

    # Create working directory that will hold all the trainig progress
    # and results.
    # In addition, dump the config file to the folder for reproducibility
    base_folder = get_base_folder(args.config_file)
    work_dir = make_abs_path(base_folder, config['work_dir'])
    create_work_dir(work_dir)
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Create data loaders
    print('Initialize dataloaders ...')
    train_dataloader, val_dataloader = \
        create_data_loaders(config['train_dataset'], config['val_dataset'], \
                            config['batch_size'])

    # Create evaluation class    
    print('Initialize evaluation class ...')
    evaluater = Evaluater(val_dataloader, k = config['val_top_k'], \
                            percentage = config['val_percentage'])

    # Initialize model
    model_name = config['model']
    use_pretrained = not config['train_from_scratch']
    print('Initialize model {} ...'.format(model_name))
    if use_pretrained:
        print('\t Use transfer learning')
    else:
        print('\t Train model from scratch')
    class_ids, _ = train_dataloader.dataset.get_classes()
    model, _ = initialize_model(model_name, len(class_ids), \
                                use_pretrained = use_pretrained)

    device = get_device(cuda = config['use_cuda'])
    print('Chosen device type: {}'.format(device.type))

    # Optimizer
    optimizer_type = config['optimizer']
    lr = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    print('Initialize optimizer {} ...'.format(optimizer_type))
    if (optimizer_type == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, \
                                    momentum=momentum, dampening=0, \
                                    weight_decay=weight_decay, nesterov=False)
    else: # Adam
        print('\t Parameter \'momentum\' is ignored')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, \
                                    betas=(0.9, 0.999), eps=1e-08, \
                                    weight_decay=weight_decay, amsgrad=False)

    # Learning rate schedulder
    print('Initialize learning rate scheduler ...')
    lr_scheduler_step_size = config['lr_scheduler_step_size']
    lr_scheduler_fac = config['lr_scheduler_fac']
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, \
                                            step_size=lr_scheduler_step_size, \
                                            gamma=lr_scheduler_fac)

    # Loss
    # Calculate loss weights dependent on label occurances
    use_loss_weights = config['use_loss_weights']
    loss_weights = calculate_loss_weights(train_dataloader.dataset) \
                    if use_loss_weights else None
    loss_func = torch.nn.CrossEntropyLoss(weight=loss_weights, \
                                        size_average=None, ignore_index=-100, \
                                        reduce=None, reduction='mean', \
                                        label_smoothing=0.0)

    # Run the training
    start = time.time()
    print('\nTraining started ...')
    num_epochs = config['num_epochs']
    val_epoch = config['val_epoch']
    model, log_train = train_model(work_dir, model, device, train_dataloader, \
                                loss_func = loss_func, optimizer = optimizer, \
                                lr_scheduler = lr_scheduler, \
                                num_epochs = num_epochs, \
                                evaluater = evaluater, \
                                eval_each_k_epoch = val_epoch,
                                plot = True, stop = None)

    end = time.time()
    print('Training took {:.2f} minutes'.format((end - start)/60.0))

    # Print the loss and eval plots
    if plot_loss(os.path.join(work_dir, 'train_log.pkl')) is not None:
        plt.savefig(os.path.join(work_dir, 'loss.png'), bbox_inches='tight')
        plt.close
    if plot_eval(os.path.join(work_dir, 'train_log.pkl')) is not None:
        plt.savefig(os.path.join(work_dir, 'eval.png'), bbox_inches='tight')
        plt.close

    print('\nTraining completed !')
    return


if __name__ == '__main__':
    main()
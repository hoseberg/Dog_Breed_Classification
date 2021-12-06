# ****************************************************************************
#  app_helpers.py
# ****************************************************************************
#
#  Author:          Horst Osberger
#  Description:     Helper functions for web app.
#
#  (c) 2021 by Horst Osberger
# ****************************************************************************

# Import others
import os
import sys
import re
import json
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objs as pgo

import torch
from torchvision.transforms import ToPILImage

# Add parent folder to include paths
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from source.functions import initialize_model, get_device, \
                             get_num_classes_from_model_state



def get_file_storage_from_request(request, tag_name):
    """
    Helper to get file storage object from flask-request.
    Args:
        request: request from client
        tag_nane: name of the requested file tag
    Returns:
        FileStorage class or None, if not available.
    """
    if tag_name in request.files:
        fs = request.files[tag_name]
        if fs.filename == '':
            # No file available
            return None
        return fs
    else:
        # No file available
        return None


def get_json_from_request(request, tag_name):
    """
    Helper to get json from file storage object.
    Args:
        request: request from client
        tag_nane: name of the requested file tag
    Returns:
        Either returns the object, an error message, or None if not requested
    """
    fs = get_file_storage_from_request(request, tag_name)
    if fs is None:
        return None

    if re.search("json", fs.content_type) is not None:
        json_read = fs.read().decode('utf8').replace("'", '"')
        return json.loads(json_read)
    else:
        return "Invalid format for json file."


def get_image_from_request(request, tag_name):
    """
    Helper to get PIL image from file storage object.
    Args:
        request: request from client
        tag_nane: name of the requested file tag
    Returns:
        Either returns the object, an error message, or None if not requested
    """
    fs = get_file_storage_from_request(request, tag_name)
    if fs is None:
        return None

    if re.search("image", fs.content_type) is not None:
        return (fs.filename, Image.open(fs))
    else:
        return "Invalid format for image file."


def check_config(config):
    """
    Helper that checks correctness of config file
    Args:
        config: config dict containing training information
    Returns:
        True, if config is as expected, False else.
    """
    # 
    # Check for required keys
    required_keys = ["model", "work_dir", "train_dataset", "val_dataset", \
                     "test_dataset", "val_top_k"]
    keys = config.keys()
    valid = True
    for key in required_keys:
        valid = valid and (key in keys)

    if not valid:
        return valid
    # 
    # Check for absolute paths
    valid = valid and os.path.isabs(config["work_dir"])
    valid = valid and os.path.isabs(config["train_dataset"])
    valid = valid and os.path.isabs(config["test_dataset"])
    valid = valid and os.path.isabs(config["val_dataset"])

    return valid


def load_model(model_name, model_state_dict_path):
    """
    Helper to load a model.
    Args:
        model_name: Name of the model
        model_state_dict_path: Path to the model state dict to be loaded
    Returns:
        Loaded model and its required input size.
    """
    state_dict = torch.load(model_state_dict_path)
    num_classes = get_num_classes_from_model_state(state_dict)
    model, input_size = initialize_model(model_name, num_classes, \
                                        use_pretrained = False)
    model.load_state_dict(state_dict)

    return model, input_size


def tensor_to_plotly(img):
    """
    Helper to transform torch.Tensor image to plotly image.
    Args:
        img: Image to be visualized
    Returns:
        Plotly graph of the image and its layout
    """
    if len(img.size()) == 4:
        img = img.squeeze(0)
    # For better display, shift to [0,1]
    img = (img - img.min())/(img.max() - img.min())
    # Ensure that tensor is on CPU
    device_cpu = get_device(cuda=False)
    img_pil = ToPILImage()(img.to(device_cpu))
    pl_img = pgo.Image(z=img_pil)
    layout = dict(title = 'Preprocessed image')
    return pl_img, layout


def train_loss_to_plotly(train_log_file):
    """
    Helper to create train loss plot.
    Args:
        train_log_file: Log-file path containing info collected during training.
    Returns:
        Plotly graph of the loss and its layout
    """
    train_log = torch.load(train_log_file)
    epochs = train_log['train_epoch']
    loss = train_log['train_loss']
    # 
    pl_loss = pgo.Scatter(x = epochs, y = loss, mode = 'lines', name = 'loss')
    layout = dict(title = 'Training Loss', 
                    xaxis = dict(title = 'Epoch'),
                    yaxis = dict(title = 'Loss value'))
    return pl_loss, layout


def train_eval_to_plotly(train_log_file, k):
    """
    Helper to create train evaluation plot.
    Args:
        train_log_file: Log-file path containing info collected during training.
        k: k of top-k error
    Returns:
        Plotly graph of the evaluation and its layout
    """
    train_log = torch.load(train_log_file)
    epochs = train_log['eval_epoch']
    top1 = train_log['eval_top1']
    topk = train_log['eval_topk']
    # 
    pl_eval = []
    pl_eval.append(pgo.Scatter(x = epochs, y = top1,
                        mode = 'lines+markers', name = 'top-1 error'))
    pl_eval.append(pgo.Scatter(x = epochs, y = topk,
                        mode = 'lines+markers', \
                        name = 'top-{} error'.format(k)))
    layout = dict(title = 'Training Evaluations', 
                    xaxis = dict(title = 'Epoch'),
                    yaxis = dict(title = 'Evaluation value'))
    return pl_eval, layout


def class_distribution_to_plotly(dataset):
    """
    Helper to create train evaluation plot.
    Args:
        dataset: dataset for which the class distribution is plotted
    Returns:
        Plotly graph of the class distribution and its layout
    """
    label_indices, label_ids, label_names = dataset.get_labels()
    df_train = pd.DataFrame({'index': label_indices, 'ids': label_ids, \
                            'names': label_names})
    class_counts = df_train.groupby(['index', 'names']).count()\
                        .sort_values(by = 'ids', ascending=False)
    label_names_counts = class_counts.ids.values

    # Use a graph since too many bars cannot be displayed nicely
    pl_class = pgo.Scatter(x = np.arange(len(class_counts.ids.index)), \
                        y = label_names_counts, \
                        mode = 'lines', name = 'Class distribution')
    layout = dict(title = 'Training Dataset Class Distribution', 
                    xaxis = dict(title = 'Classes (sorted by counts)'),
                    yaxis = dict(title = 'Count per Class'))
    return pl_class, layout
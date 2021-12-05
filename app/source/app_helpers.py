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
    Helper to get PIL image from file storage object
    Either returns the object, an error message, or None if not requested
    """
    fs = get_file_storage_from_request(request, tag_name)
    if fs is None:
        return None

    if re.search("image", fs.content_type) is not None:
        return (fs.filename, Image.open(fs))
    else:
        return "Invalid format for image file."


def tensor_to_plotly(img):
    """
    Helper to transform torch.Tensor image to plotly image
    """
    if len(img.size()) == 4:
        img = img.squeeze(0)
    # For better display, shift to [0,1]
    img = (img - img.min())/(img.max() - img.min())
    # Ensure that tensor is on CPU
    device_cpu = get_device(cuda=False)
    img_pil = ToPILImage()(img.to(device_cpu))
    pl_img = pgo.Image(z=img_pil)
    return pl_img

def check_config(config):
    """
    Helper that checks correctness of config file
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
    Helper to load a model
    """
    state_dict = torch.load(model_state_dict_path)
    num_classes = get_num_classes_from_model_state(state_dict)
    model, input_size = initialize_model(model_name, num_classes, \
                                        use_pretrained = False)
    model.load_state_dict(state_dict)

    return model, input_size

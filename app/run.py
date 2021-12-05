
# -------------- #
# Library import #
# -------------- #

# Import flask as python backend
from flask import Flask
from flask import render_template, request, redirect, jsonify

#from base64 import b64encode

# Import plotly for nice visualizations
import plotly
import plotly.graph_objs as pgo
import plotly.express as px

# Import others
import os
import sys
import re
import argparse
import matplotlib.pyplot as plt
import json
import time
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

# Add parent folder to include paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from source.functions import CustomImageDataset, Evaluater
from source.functions import transform_image, get_device, make_abs_path
from source.app_helpers import get_json_from_request, get_image_from_request, \
                            tensor_to_plotly, check_config, load_model

# Global variables
model = None
model_name = ""
class_names = None
input_size = 10
device =  get_device(cuda=False)

# -------------------- #
# Define flask backend #
# -------------------- #

# Initialize the web app as a Flask-class
# For details, see https://flask.palletsprojects.com/en/2.0.x/quickstart/ 
app = Flask(__name__)


# ---------------- #
# Helper functions #
# ---------------- #

def set_device_from_request(request, tag_name):
    """
    Helper to set global device.
    Returns either an error message, or None on success.
    """
    global device
    value = request.form.getlist(tag_name)
    request_cuda = len(value) > 0
    if (not request_cuda and device.type == "cpu")\
        or (request_cuda and device.type == "cuda"):
        # cuda request = off and cpu --> nothing do to
        return None
    else:
        # Try to get requested device
        try:
            device = get_device(cuda = request_cuda)
            return None
        except:
            return "Requested device cannot be selected"


# ------------------- #
# Register HTML pages #
# ------------------- #

# Index (main) page
@app.route('/', methods = ['GET', 'POST'])
@app.route('/index', methods = ['GET', 'POST'])
def index():  
    """
    Render main page.
    """

    return render_template('index.html')


@app.route('/', methods = ['GET', 'POST'])
@app.route('/predict', methods = ['GET', 'POST'])
def predict():  
    """
    Predict an image.
    """

    global model
    global model_name
    global input_size
    global class_names
    error_msg = None
    config = None
    img = None
    model_name_loc = model_name if model is not None else 'model not loaded'
    if request.method == 'POST':
        error_msg = set_device_from_request(request, 'use_cuda')
        config = get_json_from_request(request, 'config')
        img = get_image_from_request(request, 'image')

    # Check for errors
    if (error_msg is not None):
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)

    if type(config) == str:
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=config)    
    if type(img) == str:
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=img)

    # Load model from config if requested
    if config is not None:
        valid_config = check_config(config)
        if not valid_config:
            error_msg = 'Configuration file misses keys or has invalid keys.'
            return render_template('predict.html', model_loaded=(model is not None), \
                                    model_name=model_name_loc, error_msg=error_msg)
        try:
            # Load model
            model_name_loc = config['model']
            work_dir = config['work_dir']
            state_dict_path = make_abs_path(work_dir, 'best_model_state_dict.pt')
            model, input_size = load_model(model_name_loc, state_dict_path)
            model.eval()
            model_name = model_name_loc
            # Load dataset for class-names
            dataset = CustomImageDataset(config['val_dataset'])
            _, class_names = dataset.get_classes()
        except:
            error_msg = 'An error occured during loading the model.'
            return render_template('predict.html', model_loaded=(model is not None), \
                                    model_name=model_name_loc, error_msg=error_msg)

        # If model and dataset is loaded --> get image
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc)

    # At this point, it is assumed that the user already gave a config
    if (model is None): 
        error_msg = 'Please load a model first.'
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)

    # Check if cuda is selected
    is_cuda = None if device.type == "cpu" else device.type
    
    # Read in image and predict
    if img is None:
        error_msg = 'You must provide an image.'
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name, is_cuda=is_cuda, \
                                error_msg=error_msg)

    # At this point, we should have all --> run prediction
    try:
        img_name = img[0]
        img = ToTensor()(img[1])
        img = transform_image(width=input_size, height=input_size)(img.unsqueeze(0))
    except:
        error_msg = 'Unsupported image format.'
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name, is_cuda=is_cuda, \
                                error_msg=error_msg)

    # Shift to device and predict
    model = model.to(device)
    img = img.to(device)
    start = time.time()
    output = model(img)
    _, preds = torch.max(output, 1)
    end = time.time()

    # Return results
    exec_time = "{:.2f} ms (on {})".format(1000*(end - start), \
                                    "CPU" if device.type == "cpu" else "GPU")
    class_name = class_names[preds[0]]    
    graphs = []
    #pl_img = go.Image()
    graph_one = tensor_to_plotly(img)
    layout_one = dict(title = 'Preprocessed image')
    graphs.append(dict({'data': [graph_one], 'layout': layout_one}))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('predict.html', model_loaded=(model is not None), \
                            model_name=model_name, \
                            ids=ids, graphJSON=graphJSON, \
                            img_name=img_name, exec_time=exec_time, class_name=class_name)


# Run the actual application
# The wep app is located at http://0.0.0.0:3001/ (or localhost:3001/ for windows)
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
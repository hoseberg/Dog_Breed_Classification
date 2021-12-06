
# -------------- #
# Library import #
# -------------- #

# Import flask as python backend
from flask import Flask
from flask import render_template, request

#from base64 import b64encode

# Import plotly for nice visualizations
import plotly

# Import others
import os
import sys
import json
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# Add parent folder to include paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from source.functions import CustomImageDataset, Evaluater
from source.functions import transform_image, get_device, make_abs_path
from source.app_helpers import get_json_from_request, get_image_from_request, \
                                tensor_to_plotly, check_config, load_model, \
                                train_loss_to_plotly, train_eval_to_plotly, \
                                class_distribution_to_plotly

# Global variables
model = None
model_name = ""
train_dataset = None
test_dataloader = None
class_names = None
input_size = 10
train_log_file = None
val_k = None
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

def set_device_from_request(request):
    """
    Helper to set global device.
    Args:
        request: request from client
    Returns:
        Either an error message, or None on success.
    """
    global device
    value = request.form.getlist("use_cuda")
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


def get_requests(request):
    """
    Helper get all request information.
    Args:
        request: request from client
    Returns:
        An error message on error (None else), config and image tags
    """
    error_msg = None
    config = None
    img = None
    if request.method == 'POST':
        error_msg = set_device_from_request(request)
        config = get_json_from_request(request, 'config')
        img = get_image_from_request(request, 'image')

    # Check for errors
    if (error_msg is not None):
        return error_msg, config, img

    if type(config) == str:
        error_msg = config
        return error_msg, config, img

    if type(img) == str:
        error_msg = config
        return error_msg, config, img

    return error_msg, config, img


def get_load_model(config):
    """
    Helper to load requested model and dataset from config to global params.
    Args:
        config: config containting training info
    Returns:
        An error message on error (None else)
    """
    global model
    global model_name
    global input_size
    global class_names
    global train_dataset
    global test_dataloader
    
    error_msg = None
    if config is not None:
        valid_config = check_config(config)
        if not valid_config:
            return 'Configuration file misses keys or has invalid keys.'
        try:
            # Load datasets
            train_dataset = CustomImageDataset(config['train_dataset'])
            test_dataset = CustomImageDataset(config['test_dataset'])
            test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
            _, class_names = train_dataset.get_classes()
            # Load and set global model
            model_name_loc = config['model']
            work_dir = config['work_dir']
            state_dict_path = make_abs_path(work_dir, 'best_model_state_dict.pt')
            model, input_size = load_model(model_name_loc, state_dict_path)
            model.eval()
            model_name = model_name_loc
        except:
            return 'An error occured during loading the model.'

    return error_msg


def get_eval_data(config):
    """
    Helper to load requested evaluation data from config.
    Args:
        config: config containting training info
    Returns:
        An error message on error (None else)
    """
    global train_log_file
    global val_k
    
    error_msg = None
    if config is not None:
        valid_config = check_config(config)
        if not valid_config:
            return 'Configuration file misses keys or has invalid keys.'
        try:
            # Get data
            val_k = config['val_top_k']
            train_log_file = os.path.join(config['work_dir'], 'train_log.pkl')
        except:
            return 'An error occured during loading the model.'

    return error_msg


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
    # Reset some global params
    global model
    model = None
    global model_name
    model_name = ""
    global device
    device = get_device(cuda=False)
    return render_template('index.html')


@app.route('/evaluate', methods = ['GET', 'POST'])
def evaluate():
    """
    Evaluate a model.
    """
    global model
    global model_name
    global class_names
    global test_dataloader
    model_name_loc = model_name if model is not None else 'model not loaded'

    # Get and check user requests
    error_msg, config, _  = get_requests(request)
    if error_msg is not None:        
        return render_template('evaluate.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)
    
    # Load model and data (if rquested) and check error
    error_msg = get_load_model(config)
    error_msg_2 = get_eval_data(config)
    error_msg = error_msg if error_msg is not None else error_msg_2
    if error_msg is not None:        
        return render_template('evaluate.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)

    # At this point, it is assumed that the user already gave a config
    if (model is None): 
        error_msg = 'Please load a model first.'
        return render_template('evaluate.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)

    # Check if cuda is selected
    is_cuda = None if device.type == "cpu" else device.type

    # At this point, we should have all --> run evaluation
    # Shift to device and evaluate test dataset
    model = model.to(device)
    evaluater = Evaluater(test_dataloader, val_k, 1.0)
    start = time.time()
    accuracy, top_1_error, top_k_error = evaluater.eval(model, device)
    end = time.time()

    # Return results
    accuracy = "Accuracy: {:.4f}".format(accuracy)
    top_1_error = "Top-1 error: {:.4f}".format(top_1_error)
    top_k_error = "Top-{} error: {:.4f}".format(val_k, top_k_error)
    exec_time = "Evaluation time: {:.2f} sec (on {})".format((end - start), \
                                    "CPU" if device.type == "cpu" else "GPU")
    graphs = []
    graph_one, layout_one = train_loss_to_plotly(train_log_file)
    graph_two, layout_two = train_eval_to_plotly(train_log_file, val_k)
    graph_three, layout_three = class_distribution_to_plotly(train_dataset)
    graphs.append(dict({'data': [graph_one], 'layout': layout_one}))
    graphs.append(dict({'data': graph_two, 'layout': layout_two}))
    graphs.append(dict({'data': [graph_three], 'layout': layout_three}))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('evaluate.html', model_loaded=(model is not None), \
                            model_name=model_name, is_cuda=is_cuda, \
                            ids=ids, graphJSON=graphJSON, \
                            accuracy=accuracy, top_1_error=top_1_error, \
                            top_k_error=top_k_error, exec_time=exec_time)


@app.route('/predict', methods = ['GET', 'POST'])
def predict():  
    """
    Predict an image.
    """
    global model
    global model_name
    global input_size
    global class_names
    model_name_loc = model_name if model is not None else 'model not loaded'

    # Get and check user requests
    error_msg, config, img = get_requests(request)
    if error_msg is not None:        
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)
    
    # Load model (if rquested) and check error
    error_msg = get_load_model(config)
    if error_msg is not None:        
        return render_template('predict.html', model_loaded=(model is not None), \
                                model_name=model_name_loc, error_msg=error_msg)

    # In case of no user inputs --> just render template
    if error_msg == None and config == None and img == None:
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
        # it might be that there is a 4th (alpha) channel, just remove it
        if (img.shape[0] >= 4):
            img = img[0:3,:]
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
    graph_one, layout_one = tensor_to_plotly(img)
    graphs.append(dict({'data': [graph_one], 'layout': layout_one}))
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('predict.html', model_loaded=(model is not None), \
                            model_name=model_name, is_cuda=is_cuda, \
                            ids=ids, graphJSON=graphJSON, \
                            img_name=img_name, exec_time=exec_time, class_name=class_name)


# Run the actual application
# The wep app is located at http://0.0.0.0:3001/ (or localhost:3001/ for windows)
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
# ****************************************************************************
#  functions.py
# ****************************************************************************
#
#  Author:          Horst Osberger
#  Description:     Helper functions and classes for DL applications.
#
#  (c) 2021 by Horst Osberger
# ****************************************************************************

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_files       

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
import torchvision.models as models


def get_base_folder(file_path = None):
    """
    Helper to get the correct base folder for relative paths.
    This might be required since our CustomImageDataset requires absolut paths,
    however, paths in a config file might be relative (to the config itself).
    If no (config) file is given, it can be assumed that paths are absolute or
    relative to where the script is called from.

    Args: 
        file_path: Path to a file (e.g., config file)

    Returns:
        Either the parent's absolute path of the file_path, or os.getcwd()
    """
    if file_path is None:
        return os.getcwd()

    # Get parent folder of file
    if not os.path.isfile(file_path):
        raise Exception('Param {} must be a file, but is not'.format(file_path))

    file_path = os.path.abspath(file_path)
    return os.path.split(file_path)[0]


def make_abs_path(base_folder, file_path):
    """
    Helper script to return correct absolut path

    Args:
        file_path: Path to a file
        base_folder: Excepted folder the file_path is relative to

    Returns:
        Absolute file path of file_path
    """
    if (not os.path.isabs(file_path)):
        file_path = os.path.join(base_folder, file_path)

    return file_path


def import_data(image_dir):
    """
    From a given directory, load all image files and identify class labels.
    It is assumed that for each class, a separate folder exists that has either
    the form
        <class_name>
    or 
        <class_id.class_name>
    In the first case, a class id is assigned from 0 to #(class_names)-1 for. 
    each class. In the second case, the folder name is splitted and the first
    part is chosen as class id whereas the second one is chosen as class name.

    Args:
        image_dir: Folder containing all the images to be loaded
    Returns:
        class_ids:      List of (unique) class ids
        class_names:    List of (unique) class names
        file_path:      List of all file paths
        label_index:    List of indices to match entries in file_path with
                        label ids and label names. Same length as file_path.
        label_id:       List of label ids. Same length as file_path.
        label_name:     List of label names. Same length as file_path.
    """
    # 
    # Let us use sklearn's datasets.load_files() function. This function
    # returns a data struct/bundle with index-like target, filenames, 
    # and target name.
    #   target_names... Corresponds to class names, hence is a 
    #                   sorted list with unique names of the labels.
    #                   Length = Number of classes
    #   target... Gives the position of each label in the filenames array.
    #             Length = Number of total images
    #   filenames... Names of the files. Length = Number of total images
    data = load_files(image_dir, load_content = False)
    targets = data['target']
    target_names = data['target_names']
    file_path = data['filenames']
    # 
    # For each filename, get the label id and label name
    label_id = np.zeros(len(targets)).astype(int)
    label_name = np.zeros(len(targets)).astype(str)
    for i, target in enumerate(targets):
        target_name = target_names[target]
        item_split = target_name.split('.')
        if len(item_split) == 1:
            item_id = i
            item_label = target_name
        else:
            # Try to exract class ids and class names from folder.
            # In this case it is assumed that the info is given in the format 
            #    <class_id>.<class_name>
            try:
                item_id = int(item_split[0])
                item_label = '.'.join(item_split[1:])
            except:
                raise ValueError('Image folder does not have valid format. '
                                'Expecting \'id.name_of_class\'.')
                
        label_id[i] = item_id
        label_name[i] = item_label

    # Now, we have for each file a label and a label_id (that has not to be
    # in range 0, 1, ... !!). For later training, we need to guarantee that 
    # each class (in unqiue(label_name)) is assigned to a value in
    # range(len(label_id)). This is done here:
    sort_indices = np.argsort(label_id)
    class_ids = np.unique(label_id[sort_indices])
    # 
    class_names = np.zeros(len(class_ids)).astype(str)
    # class_indices is simply given by np.arange(len(class_names))
    label_index = np.zeros(len(label_id)).astype(int)
    for i, id in enumerate(class_ids):
        class_names[i] = [label_name[i_] for i_, x_ in enumerate(label_id) \
                                                    if x_ == id][0]
        label_index[label_id == id] = i

    return class_ids, class_names, file_path, label_index, label_id, label_name


def transform_image(width = 224, height = 224, mean = [0.485, 0.456, 0.406], \
                    std = [0.229, 0.224, 0.225]):
    """
    Transformation function for dataloader. 
    Default values are chosen to fit imagenet pretrained models.

    Args:
        width:  Required image width
        height: Required image height
        mean:   Tuple, for each channel containing required mean
        std:    Tuple, for each channel containiner required standard deviation

    Returns:
        Transformations to scale image to [0,1], normalize, and resize
        it to [3, height, width]
    """
    # Image preprocessing transform.
    preprocessing = []
    preprocessing.append(transforms.Resize((height, width)))
    # ConvertImageDtype also put range to [0, 1] !
    preprocessing.append(transforms.ConvertImageDtype(torch.float32))
    preprocessing.append(transforms.Normalize(mean, std))

    return transforms.Compose(preprocessing)


def augment_image(width = 224, height = 224, augment_prop=0.0):
    """
    Randomly augment an image.

    Args:
        width:          Required image width
        height:         Required image height
        augment_prop:   float in [0,1], gives probability for augmentation

    Returns:
        Augmented image
    """
    # Image preprocessing transform.
    preprocessing = []
    if (augment_prop > 0.0):
        preprocessing.append(transforms.RandomHorizontalFlip(p=augment_prop))
        # Random rotation does not have a random state...
        if (augment_prop > np.random.random(1)[0]):
            degrees = 5.0
            preprocessing.append(transforms.RandomRotation(
                degrees=degrees, center=[height/2, width/2],
                expand=False, fill=0))

    return transforms.Compose(preprocessing)


class CustomImageDataset(Dataset):
    """
    Child class of Dataset for custom datasets.
    """
    def __init__(self, img_dir, transform = transform_image(), augment = None,\
                 only_first_n_samples = None, cache_size = 0.0):
        """
        Initialize the custom dataset.

        Args:
            img_dir:    Image directory as required for import_data()
            transform:  Transformations to preprocess raw sample for models
            augment:    Augmentation functions
            only_first_n_samples: If set, only first n samples are selected
                                  (For testing issues)
            cache_size: In GB. If set >0.0, this is the maximum memory that
                        can be used for caching data. Only usable if Dataloader
                        that uses this class has num_workers = 0.
        """
        # 
        # We require absolut path
        if not os.path.isabs(img_dir):
            raise ValueError('Image dir must be absolute path, but is {}'\
                            .format(img_dir))
        # 
        # Load the data
        class_ids, class_names, file_path, label_index, label_id, label_name \
            = import_data(img_dir)
        # 
        self.img_dir = img_dir
        self.class_ids = class_ids
        self.class_names = class_names
        self.file_path = file_path
        self.label_index = label_index
        self.label_id = label_id
        self.label_name = label_name
        self.transform = transform
        self.augment = augment
        self.only_first_n_samples = only_first_n_samples
        # 
        self.cache_size = cache_size
        self.use_cache = cache_size > 0.0
        self.cache_initialized = False
        self.cache = None
        self.cache_index_list = []

    def __len__(self):
        """
        Returns:
            Number of samples in the dataset.
        """
        if (type(self.only_first_n_samples) == type(1)):
            return min(self.only_first_n_samples, len(self.file_path))
        else:
            return len(self.file_path)

    def __getitem__(self, idx):
        """
        This function must return samples in CHW format.

        Args:
            idx: Index of the requested sample
        Returns:
            Batch of images (image) and the associated label indices (idx)
        """
        # 
        if (self.use_cache and self.cache_initialized \
            and idx in self.cache_index_list):
            image = self.cache[idx]
        else:
            # Read image from file and apply transform
            # 
            # It might be that there are erroneous images in the dataset
            # --> try to skip these and take neighbor images instead
            success = False
            num_tries = 0
            idx_orig = idx
            while (not success and num_tries < 4):
                try:
                    img_path = self.file_path[idx]
                    image = read_image(img_path)
                    success = True
                except Exception:
                    num_tries += 1
                    idx = max(0, idx - num_tries) if num_tries%2 == 0 \
                                else min(len(self.file_path), idx + num_tries)

            if (not success):
                raise ValueError('Image at position {} could not be read'\
                                .format(idx_orig))
            
            # We require 3-channel CHW real images
            if (len(image.shape) != 3 and len(image.shape) != 4):
                raise ValueError('Image dimension must be 3 or 4, but is {}'\
                                .format(image.shape))
            # In case of multiy-channel images (with alpha) --> reduce 
            if (len(image.shape) == 3 and image.shape[0] >= 4):
                image = image[0:3, :]
            if (len(image.shape) == 4 and image.shape[1] >= 4):
                image = image[:, 0:3, :]

            # Apply transformation
            image = self.transform(image)
            # Add to cache...
            if (self.use_cache and self.cache_initialized \
                    and idx < self.cache.shape[0]):
                self.cache[idx] = image
                self.cache_index_list.append(idx)

        # Init cache.
        # For the very first call, this is the place where we know the shape
        # of our tensors --> init cache
        # The cache_size is in GB--> calls how many samples we can cache
        if (self.use_cache and not self.cache_initialized):
            shape = image.shape
            cxhxw = shape[0]*shape[1]*shape[2]
            cache_num_samples = int((1e9 * self.cache_size) / (4*cxhxw))
            cache_num_samples = min(self.__len__(), cache_num_samples)
            self.cache = torch.Tensor(np.zeros([cache_num_samples, shape[0], \
                                        shape[1], shape[2]], dtype=float))
            self.cache_initialized = True

        # Apply random augmentation, if existing
        if (self.augment is not None):
            image = self.augment(image)
        
        # We do not return the label index, but the index of where
        # the image occures in the dataset. The correct label name/id/index
        # can be retrieved from the self.label_ tuples
        return image, idx
    
    def get_classes(self):
        """
        Returns:
            List of class names
        """
        return self.class_ids, self.class_names

    def get_labels(self):
        """
        Returns:
            Label indices, ids, and names as described in import_data() 
        """
        return self.label_index, self.label_id, self.label_name
    
    def get_file_path(self):
        """
        Returns:
            List of file paths
        """
        return self.file_path
    
    def set_preprocessed_data(self, data):
        """
        Set preprocessed data as a torch.Tensor that can be used for data
        caching.
        Attention: Data is not checked!
        """
        if (type(data) is not torch.Tensor):
            raise ValueError('data must be torch.Tensor')

        self.cache = data
        self.use_cache = True
        self.cache_initialized = True
        self.cache_index_list = list(range(0, data.shape[0]))
        return


def get_device(cuda = True):
    """
    Get compute device.

    Args:
        cuda: If set to True, a CUDA capable GPU is requested
    Returns:
        Either a CPU or GPU device.
    """
    if cuda and not torch.cuda.is_available():
        raise ValueError("Requested cuda device is not available")

    device = torch.device("cuda:0" if cuda else "cpu")
    return device


def initialize_model(model_name, num_classes, use_pretrained=True):
    """
    Initialize models from the pytorch model zoo.
    All pre-trained models expect input images normalized in the same 
    way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    where H and W are expected to be at least 224. The images have to 
    be loaded in to a range of [0, 1] and then normalized using 
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        model_name: Model name requested from the model zoo
        num_classes: Number of classes set to the model
        use_pretrained: If True, pretrained weigths (pretrained on ImageNet)
                        are downloaded and set to the model.
    Returns:
        Requested model
    """
    model = None
    input_size = 0

    if model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        has_bias = model.fc.bias is not None
        model.classifier[6] = nn.Linear(num_ftrs,num_classes, has_bias)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model.classifier.in_features
        has_bias = model.classifier.bias is not None
        model.classifier = nn.Linear(num_ftrs, num_classes, has_bias)
        input_size = 224

    elif model_name == "resnet18":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        has_bias = model.fc.bias is not None
        model.fc = nn.Linear(num_ftrs, num_classes, has_bias)
        input_size = 224
        
    elif model_name == "resnet50":
        """ Resnet50
        """
        model = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        has_bias = model.fc.bias is not None
        model.fc = nn.Linear(num_ftrs, num_classes, has_bias)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model.classifier[6].in_features
        has_bias = model.classifier[6].bias is not None
        model.classifier[6] = nn.Linear(num_ftrs,num_classes, has_bias)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), \
                                        stride=(1,1))
        model.num_classes = num_classes
        input_size = 224

    else:
        raise ValueError("Invalid model name, exiting...")

    return model, input_size


def get_num_classes_from_model_state(model_state_dict):
    """
    Get the number of classes from a model state dict. This function only
    supports models as covered in initialize_model(). It makes use of the fact
    that the last layer is either a fc-layer with #bias=num_classes, or a
    convolution layer with #kernels=num_classes. 

    Args:
        model_state_dict: State dictionary of a model

    Returns:
        Number of classes the model was trained for
    """
    last_key = list(model_state_dict.keys())[-1]    
    return model_state_dict[last_key].shape[0]


def train_model(work_dir, model, device, train_dataloader, \
                loss_func, optimizer, lr_scheduler, num_epochs=25, \
                evaluater = None, eval_each_k_epoch = 1,
                plot = False,
                stop = None):
    """
    Train a model.

    Args:
        work_dir:           Working directory. All training info is saved to
                            this directory.
        model:              Model to be trained
        device:             Device on which the model is trained
        train_dataloader:   Dataloader that serves for training inputs/targets
        loss_func:          Loss used for training
        optimizer:          Optimizer used for optimization
        lr_scheduler:       Learning rate scheduling
        num_epochs:         Number of epochs to be trained
        evaluater:          Evaluation class to evaluate intermediate model
        eval_each_k_epoch:  At each k-th epoch, the evaluation is done
        plot:               If True, results are printed to console
        stop:               Lambda function that can be used for interruption
    Returns:
        Trained model and training log.
    """
    #
    # Initialize the log dict for training
    # This dict keeps the main information of the training progress. 
    log_train = dict({'num_epochs': num_epochs, \
                      'train_epoch': [], 'train_loss': [], \
                      'eval_epoch': [], 'eval_top1': [], 'eval_topk': []})
    
    # Get label information from training loaders
    label_indices, _, _ = train_dataloader.dataset.get_labels()
        
    # Shift model to device
    model = model.to(device)
    loss_func = loss_func.to(device)
    
    # We will average the loss over a certain number of samples
    batch_size = train_dataloader.batch_size
    num_batches_per_epoch = int(len(train_dataloader.dataset) / batch_size)
    loss_avg_n = batch_size * max(1, int(num_batches_per_epoch/3))
    iteration = 0
    iterations_per_epoch = len(train_dataloader.dataset)
    running_iters = 0
    running_loss = 0.0
    
    # Run the training
    for epoch in range(num_epochs):
        if plot:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        
        # Set model to training mode
        model.train()

        # Iterate over data.
        for batch, label_idx in train_dataloader:
            # 
            # In case the process is stopped
            if (stop is not None and stop()):
                break
            # 
            # Get correct labels indices
            labels = label_indices[label_idx]

            # Shift data to device
            batch = batch.to(device)
            labels = torch.Tensor(label_indices[label_idx])
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward and backward propagation
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(batch)
                loss = loss_func(outputs, labels.long())
                loss.backward()
                optimizer.step()

            # statistics
            running_iters += batch.size(0)
            running_loss += loss.item() * batch.size(0)
            
            iteration_new = iteration + batch.size(0)
            if (int(iteration/loss_avg_n) < int(iteration_new/loss_avg_n)):
                # Save the loss
                curr_epoch = iteration_new / iterations_per_epoch
                curr_loss = running_loss / max(1, running_iters)
                log_train['train_epoch'].extend([curr_epoch])
                log_train['train_loss'].extend([curr_loss])
                if plot:
                    print('Epoch {:.2f}, Loss: {:.4f}'\
                            .format(curr_epoch, curr_loss))
                running_loss = 0.0
                running_iters = 0
                # Save to file
                torch.save(log_train, work_dir + '/train_log.pkl')
                
            iteration = iteration_new
        
        # 
        # In case the process is stopped
        if (stop is not None and stop()):
            break            
            
        # After each epoch, update the scheduler
        lr_scheduler.step()
        
        # Do the evaluation
        if (evaluater is not None and (epoch%eval_each_k_epoch == 0)):
            _, top_1_error, top_k_error = evaluater.eval(model, device)
            log_train['eval_epoch'].extend([float(epoch)])
            log_train['eval_top1'].extend([top_1_error])
            log_train['eval_topk'].extend([top_k_error])
            if plot:
                print('EVAL: Epoch {}, top_1: {:.4f}, top_k: {:.4f}'\
                        .format(epoch+1, top_1_error, top_k_error))
            # Save to file
            torch.save(log_train, work_dir + '/train_log.pkl')
        
        # Save initial model and if it has improved
        top_l_errors = log_train['eval_top1']
        if (len(top_l_errors) == 1 or \
            (len(top_l_errors) > 2 and (top_l_errors[-1] < top_l_errors[-2]))):
            torch.save(model.state_dict(), \
                        work_dir + '/best_model_state_dict.pt')
    
    return model, log_train


class Evaluater():
    """
    Evaluation class.
    """
    def __init__(self, dataloader, k, percentage = 1.0):
        """
        Initialize evaluation class.
        Args:
            dataloader: Dataloader the evaluation runs on
            k:          k for top-l error
            percentage: Percentage how many samples should be used for
                        evaluation
        """
        self.dataloader = dataloader
        self.k = k
        self.percentage = percentage
        
    def eval(self, model, device):
        """
        Run evaluation on a model and a device.
        Args:
            model:  Model to be evaluated
            device: Device the evaluation should run on
        """
        # Set model to evaluate mode
        model.eval()
        model = model.to(device)
        
        batch_size = self.dataloader.batch_size
        preds_top1 = np.zeros(batch_size*len(self.dataloader)).astype(bool)
        preds_topk = np.zeros(batch_size*len(self.dataloader)).astype(bool)
        
        label_indices, _, _ = self.dataloader.dataset.get_labels()
        
        for i, (batch, label_idx) in enumerate(self.dataloader):
            # 
            if (i > self.percentage*len(self.dataloader)):
                preds_top1 = preds_top1[0:i*batch_size]
                preds_topk = preds_topk[0:i*batch_size]
                break
            # 
            # Get correct labels indices.
            # Attention, if batch_size=1, label_idx is no tuple!
            tmp = label_indices[label_idx]
            tmp = tmp if tmp.size > 1 else [tmp]
            labels = torch.Tensor(tmp)
            labels = labels.to(dtype=torch.int)

            # Shift data to device
            batch = batch.to(device)
            labels = labels.to(device)
            
            # Evaluate batch
            with torch.set_grad_enabled(False):
                outputs = model(batch)
                #_, preds = torch.max(outputs, 1)
                # 
                # Get top k results for each batch item
                tmp_preds_top1 = np.zeros(outputs.shape[0])
                tmp_preds_topk = np.zeros(outputs.shape[0])
                num_samples = outputs.shape[0]
                for b in range(0, num_samples):
                    # _, preds = torch.max(outputs, 1)
                    top_k = outputs[b].argsort(descending=True)[0:self.k]
                    top_1 = top_k[0:1]
                    tmp_preds_top1[b] = (labels[b] in top_1)
                    tmp_preds_topk[b] = (labels[b] in top_k)
                
                preds_top1[i*num_samples:(i+1)*num_samples] = tmp_preds_top1
                preds_topk[i*num_samples:(i+1)*num_samples] = tmp_preds_topk
                
        # The dataloader maybe filled batches at the end
        if (len(preds_top1) > len(self.dataloader.dataset)):
            preds_top1 = preds_top1[0:len(self.dataloader.dataset)]
            preds_topk = preds_topk[0:len(self.dataloader.dataset)]
            
        accuracy = sum(preds_top1)/len(preds_top1)
        top_1_error = 1.0 - accuracy
        top_k_error = 1.0 - sum(preds_topk)/len(preds_top1)
        
        return accuracy, top_1_error, top_k_error


def plot_loss(train_log_file):
    """
    Create the plot for the loss.
    Args:
        train_log_file: Path to training log containing training info
    Returns:
        Matplotlib pyplot, or None, if no log gile is given
    """
    # 
    try:
        train_log = torch.load(train_log_file)
    except:
        # no file, just return None
        return None

    epochs = train_log['train_epoch']
    loss = train_log['train_loss']
    # 
    fig = plt.figure()
    fig.suptitle('Training loss', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.plot(epochs, loss)
    
    return (fig, ax)


def plot_eval(train_log_file):
    """
    Create the plot for the evaluation.
    Args:
        train_log_file: Path to training log containing training info
    Returns:
        Matplotlib pyplot, or None, if no log gile is given
    """
    # 
    try:
        train_log = torch.load(train_log_file)
    except:
        # no file, just return None
        return None

    epochs = train_log['eval_epoch']
    top1 = train_log['eval_top1']
    topk = train_log['eval_topk']
    # 
    fig = plt.figure()
    fig.suptitle('Evaluation', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Top k errors')
    ax.plot(epochs, top1, label='Top-1 error')
    ax.plot(epochs, topk, label='Top-k error')
    ax.legend()
            
    return (fig, ax)


def calculate_loss_weights(dataset):
    """
    Calculate loss weights. Loss weigths are used to increase the impact of
    samples from classes that appear less in the dataset than other classes.

    Args:
        train_dataloader: Dataset from which the class distribution is chosen

    Returns:
        Loss weights according to the dataset (as torch.Tensor)
    """
    # 
    # Create a pandas Dataframe to get class counts
    label_indices, label_ids, label_names = dataset.get_labels()
    df_train = pd.DataFrame({'index': label_indices, 'ids': label_ids, \
                             'names': label_names})
    class_counts = df_train.groupby(['index', 'names']).count()
    label_names_counts = class_counts.ids.values
    loss_weights = min(label_names_counts)/label_names_counts
    loss_weights = torch.Tensor(loss_weights)

    return loss_weights
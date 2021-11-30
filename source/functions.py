import numpy as np
from glob import glob
import os
from numpy.core.fromnumeric import squeeze
#import pandas as pd
import math
import time

from IPython import display

from PIL import Image
import matplotlib.pyplot as plt

from sklearn.datasets import load_files       

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torch.nn.functional import interpolate
import torch.nn as nn

from torchvision import transforms
from torchvision.io import read_image
import torchvision.models as models


def import_data(image_dir):
    # 
    # load_files returns a data struct/bundle with 
    # index-like target, filenames, and target name.
    #       target_names... corresponds to class names, hence is a 
    #                       sorted list with unique names of the labels.
    #                       length = Number of classes
    #       targets...  gives the position of each label in the filenames array
    #                   length = Number of total images
    #       filenames... 
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
            # Try to exract label ids and label/class names from folder.
            # In this case it is assumed that the label is given in the format 
            #    <label_id>.<label_name>
            try:
                item_id = int(item_split[0])
                item_label = '.'.join(item_split[1:])
            except:
                raise ValueError('Image folder does not have valid format. Expecting \'id.name_of_class\'.')
                
        label_id[i] = item_id
        label_name[i] = item_label

    # Now, we have for each file a label and a label_id (that has not to be in range 0, 1, ... !!).
    # For later training, we need to guarantee that each class (in unqiue(label_name)) is assigned
    # to a value in range(len(label_id)). This is done here:
    sort_indices = np.argsort(label_id)
    class_ids = np.unique(label_id[sort_indices])
    # 
    class_names = np.zeros(len(class_ids)).astype(str)
    # class_indices is simply given by np.arange(len(class_names))
    label_index = np.zeros(len(label_id)).astype(int)
    for i, id in enumerate(class_ids):
        class_names[i] = [label_name[i_] for i_, x_ in enumerate(label_id) if x_ == id][0]
        label_index[label_id == id] = i

    return class_ids, class_names, file_path, label_index, label_id, label_name


def transform_image(width = 224, height = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """
    Transformation function for dataloader. 
    Default values are chosen to fit imagenet pretrained models.

    Args:
        width: required image width
        height: required image height
        mean: tuple, for each channel containing required mean
        std: tuple, for each channel containiner required standard deviation

    Returns:
        transformations for image scaled to [0,1], normalized, and resized to [3, hight, width]
    """
    # Image preprocessing transform.
    preprocessing = []

    preprocessing.append(transforms.Resize((height, width)))
    # ConvertImageDtype also put range to [0, 1] !
    preprocessing.append(transforms.ConvertImageDtype(torch.float32))
    preprocessing.append(transforms.Normalize(mean, std))

    return transforms.Compose(preprocessing)

    """
    # interpolate requires 4d tensors
    squeeze_shape = len(image.shape) == 3
    if (squeeze_shape):
        image = image.unsqueeze(0)
    image = interpolate(image, size=[width, height], mode="nearest", align_corners=False)
    if (squeeze_shape):
        image = image.squeeze(0)

    # type conversion
    dtype_orig = image.dtype
    if image.dtype != torch.float32:
        image = transforms.ConvertImageDtype(torch.float32)(image)

    # Ensure that range is in [0, 1]. Note that
    # transforms.ConvertImageDtype already scales to [0, 1] !
    if (image.min() < 0.0 or image.max() > 1.0):
        if dtype_orig == torch.uint8:
            # Byte [0, 255] --> float [0, 1]
            image = image.multiply(1.0/255.0)
        else:
            image = (image - image.min())/(image.max() - image.min())

    # normalization
    image = transforms.Normalize(mean, std)(image)
    
    return image
    """


def augment_image(width = 224, height = 224, augment_prop=0.0):
    """
    Randomly augment an image

    Args:
        augment_prop: float in [0,1], gives probability for augmentation

    Returns:
        augmented image
    """
    # Image preprocessing transform.
    preprocessing = []
    if (augment_prop > 0.0):
        preprocessing.append(transforms.RandomHorizontalFlip(p=augment_prop))
        # Random rotation does not have a random state...
        if (augment_prop > np.random.random(1)[0]):
            degrees = 5.0
            # Add symmetric padding to avoid boarder issues
            #padding = int(height*np.tan(math.radians(degrees)))
            #preprocessing.append(transforms.Pad(padding, padding_mode='symmetric'))
            preprocessing.append(transforms.RandomRotation(
                degrees=degrees, center=[height/2, width/2],
                expand=False, fill=0))
            # Back to correct image size
            #preprocessing.append(transforms.Resize((height, width)))

    return transforms.Compose(preprocessing)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform = transform_image(), augment = None,\
                 only_first_n_samples = None, cache_size = 0.0):
        # 
        # We require absolut path
        if not os.path.isabs(img_dir):
            raise ValueError('Image dir must be absolute path, but is {}'.format(img_dir))
        # 
        # Load the data
        class_ids, class_names, file_path, label_index, label_id, label_name = import_data(img_dir)
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
        if (type(self.only_first_n_samples) == type(1)):
            return min(self.only_first_n_samples, len(self.file_path))
        else:
            return len(self.file_path)

    def __getitem__(self, idx):
        """
        This function must return CHW format
        """
        # 
        if (self.use_cache and self.cache_initialized and idx in self.cache_index_list):
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
                raise ValueError('Image at position {} could not be read'.format(idx_orig))
            
            # We require CHW real images
            if (len(image.shape) != 3 and len(image.shape) != 4):
                raise ValueError('Image dimension must be 3 or 4, but is {}'.format(image.shape))
        
            # Apply transformation
            image = self.transform(image)
            # Add to cache...
            if (self.use_cache and self.cache_initialized and idx < self.cache.shape[0]):
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
        return self.class_ids, self.class_names

    def get_labels(self):
        return self.label_index, self.label_id, self.label_name
    
    def get_file_path(self):
        return self.file_path
    
    def set_preprocessed_data(self, data):
        """
        Set preprocessed data as a torch.Tensor.
        Data is not checked! 
        """
        if (type(data) is not torch.Tensor):
            raise ValueError('data must be torch.Tensor')

        self.cache = data
        self.use_cache = True
        self.cache_initialized = True
        self.cache_index_list = list(range(0, data.shape[0]))
        return


def get_device(cuda = True):
    if cuda and not torch.cuda.is_available():
        raise ValueError("Requested cuda device is not available")

    device = torch.device("cuda:0" if cuda else "cpu")
    return device


def initialize_model(model_name, num_classes, use_pretrained=True):
    """
    All pre-trained models expect input images normalized in the same 
    way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), 
    where H and W are expected to be at least 224. The images have to 
    be loaded in to a range of [0, 1] and then normalized using 
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
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
        has_bias = model.fc.bias is not None
        model.classifier = nn.Linear(num_ftrs, num_classes, has_bias)
        input_size = 224

    elif model_name == "googlenet":
        """ Googlenet
        """
        model = models.googlenet(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        has_bias = model.fc.bias is not None
        model.fc = nn.Linear(num_ftrs, num_classes, has_bias)
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
        has_bias = model.fc.bias is not None
        model.classifier[6] = nn.Linear(num_ftrs,num_classes, has_bias)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224

    else:
        raise ValueError("Invalid model name, exiting...")

    return model, input_size


def train_model(work_dir, model, device, train_dataloader, \
                loss_func, optimizer, lr_scheduler, num_epochs=25, \
                evaluater = None, eval_each_k_epoch = 1,
                plot = False,
                stop = None):
    """
    TODO: Write doc
    """
    #
    # Initialize the lod dict for training
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
                    print('Epoch {:.2f}, Loss: {:.4f}'.format(curr_epoch, curr_loss))
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
                print('EVAL: Epoch {:.2f}, top_1: {:.4f}, top_k: {:.4f}'.format(float(epoch), top_1_error, top_k_error))
            # Save to file
            torch.save(log_train, work_dir + '/train_log.pkl')
        
        # Save initial model and if it has improved
        top_l_errors = log_train['eval_top1']
        if (len(top_l_errors) == 1 or \
            (len(top_l_errors) > 2 and (top_l_errors[-1] < top_l_errors[-2]))):
            torch.save(model.state_dict(), work_dir + '/best_model_state_dict.pt')
    
    return model, log_train


class Evaluater():
    def __init__(self, dataloader, k, percentage = 1.0):
        self.dataloader = dataloader
        self.k = k
        self.percentage = percentage
        
    def eval(self, model, device):
        
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
            # Get correct labels indices
            labels = torch.Tensor(label_indices[label_idx])
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
                for b in range(outputs.shape[0]):
                    top_k = outputs[b].argsort()[0:self.k]
                    top_1 = top_k[0:1]
                    tmp_preds_top1[b] = (labels[b] in top_1)
                    tmp_preds_topk[b] = (labels[b] in top_k)
                
                preds_top1[i*batch_size:(i+1)*batch_size] = tmp_preds_top1
                preds_topk[i*batch_size:(i+1)*batch_size] = tmp_preds_topk
                
        # The dataloader maybe filled batches at the end
        if (len(preds_top1) > len(self.dataloader.dataset)):
            preds_top1 = preds_top1[0:len(self.dataloader.dataset)]
            preds_topk = preds_topk[0:len(self.dataloader.dataset)]
            
        accuracy = sum(preds_top1)/len(preds_top1)
        top_1_error = 1.0 - accuracy
        top_k_error = 1.0 - sum(preds_topk)/len(preds_top1)
        
        return accuracy, top_1_error, top_k_error


def print_loss(work_dir, sleep_time = 0, stop = None):
    # 
    while (True):
        if (stop is not None and stop()):
            break

        time.sleep(sleep_time)

        try:
            train_log = torch.load(work_dir + '/train_log.pkl')
        except:
            # no file
            if stop is None:
                return
            else:
                continue

        epochs = train_log['train_epoch']
        loss = train_log['train_loss']
        # 
        plt.plot(epochs, loss, '-r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        #plt.xticks(rotation=45, ha='right')
        plt.suptitle('Training loss')
        #plt.show()
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if stop is None:
            return
    
    return

def print_eval(work_dir, sleep_time = 0, stop = None):
    # 
    while (True):
        if (stop is not None and stop()):
            break

        time.sleep(sleep_time)

        try:
            train_log = torch.load(work_dir + '/train_log.pkl')
        except:
            # no file
            if stop is None:
                return
            else:
                continue

        epochs = train_log['eval_epoch']
        top1 = train_log['eval_top1']
        topk = train_log['eval_topk']
        # 
        plt.plot(epochs, top1)
        plt.plot(epochs, topk)
        plt.xlabel('Epochs')
        plt.ylabel('Top k errors')
        #plt.xticks(rotation=45, ha='right')
        plt.suptitle('Training evaluation')
        #plt.show()
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if stop is None:
            return
    
    return
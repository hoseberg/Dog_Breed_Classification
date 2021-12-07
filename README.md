# Dog Breed Classification

Author: Horst Osberger

This is a project for the UDACITY Nanodegree "Data Scientist"


## Table of content
* [Motivation](#Chap1)
* [Description of project](#Chap2)
* [File descriptions](#Chap3)
* [Description of repository and scripts](#Chap4)
* [Installation and application instructions](#Chap5)
* [Dependencies](#Chap6)
* [Dog Breed Classificaiton -- Results](#Chap7)
* [Project conclusion](#Chap8)
* [License](#Chap9)


## Motivation <a name=Chap1></a>

For the final capstone project of the UDACITY Nanodegree "Data Scientist" course, it was free to choose a project one 
is interested in. Although, there are many projects that I'm interested, I choosed a pretty basic one, that is solving 
a Deep Learning classification problem.  
For me, the main reason for this choice was not the Deep Learning aspect of the project. Since I'm working in the Deep Learning
team of a computer vision company for several years, I'm familiar with some Deep Learning algorithms.  
However, since we developed our own Deep Learning framework in this company, I often had the feeling that I have a 
knowledge gap for other, more common Deep Learning frameworks. Therefore, I used this project to dive in more deeply
into a widely used framework... that is pytorch.

## Description of project<a name=Chap2></a>

In this python project, a simple Deep Learning classification framework is build based on pytorch. While all the evaluations below are
done on the so-called [Dog Breed Dataset available here](https://github.com/udacity/dog-project), the scripts can be used
for other classification tasks as well.

However, in the following, we will always consider the Dog Breed Dataset for deeper discussions. 

Here, you see an example image... look how cude the dogs are!! 

![Dog sample](./screenshots/dog_example.png?raw=true "Dog sample")

The dataste contains 133 dog categories (classes) with a total number of 8351 images. The dataset is divided in a 
`training`, `validation`, and `test` dataset split, each containing 6680, 835, and 836 images, respectively. 

As you can see in the following image, the distribution of the dog breed classes in the training dataset is not even.

![Class distribution](./screenshots/class_distribution.png?raw=true "Class distribution")

Therefore, the below experiments use `class weighting`. This is a method that tries to correct inbalancing in the training dataset. 

In this project, a number of models and options can be chosen for training. Currently supported models are:
* [densenet](https://arxiv.org/abs/1608.06993)
* [resnet18](https://arxiv.org/abs/1512.03385)
* [resnet50](https://arxiv.org/abs/1512.03385)
* [squeezenet](https://arxiv.org/abs/1602.07360)
* [vgg](https://arxiv.org/abs/1409.1556)

The goal of this project is to answer the following questions:
* Is transfer-learning better than training from scratch?
* Which model performs best on the dataset?

To answer this questions, the training runs on all of the provided models using transfer-learning and learning from scratch. 
Since we use the same parameter set for all experiments, one can compare the performance between different model types, as 
well as if the transfer-learning method is better than training from scratch. See the results in the chapter 
[Dog Breed Classificaiton -- Results](#Chap7). Note that **no** parameter tuning is done here. 

To train the models properly, the data from the dataset (note that all images have different image dimensions) must
be put to a common format, hence preprocessed. All models above are selected from the
[pytroch model zoo](https://pytorch.org/vision/stable/models.html) and require the same input dimensions that is
$[3, 224, 224]$. In addition, the images must be real images. Furthermore, the models from the model zoo have been pretrained on the
[ImageNet](https://image-net.org/) dataset, and assume the input images to be normalized according to this dataset. 

Therefore, for preprocessing, the images run throw the following transformation steps:
* Resize image to $[3, 224, 224]$.
* Convert image type to real images.
* Normalize image using mean $[0.485, 0.456, 0.406]$ and standard deviation $[0.229, 0.224, 0.225]$.

In addition, the samples that are used for training are also randomly augmented. Augmentation is a method to increase 
the dataset's variations, which leads to better model generalization during the training. The following random augmentations
are performed:
* Horizontal flips
* Random rotations between $[-5.0, 5.0]$ degrees

For evaluation of the model's performance, we use the following metrics:
* accuracy
* top-1 error
* top-3 error

The accuracy and the top-1 error are closely related and give a measure about how many samples have been predicted correctly. 
In addition, the top-3 error is evaluated. The motivation behind this is, that there are many classes of different dog breeds
that look very similar, even for human eyes. Therefore, it makes sense to check if the model is at least somehow close at a
certain class. In more detail, a prediction of a sample is counted as correct if the correct class is contained in the
top-3 predictions of the model.


## File descriptions <a name=Chap3></a>

├── app\
│ ├── static\
│ │ ├── logos\
│ │ │ ├─ githublogo.png\
│ │ │ └─ linkedinlogo.png\
│ │ \
│ ├── source\
│ │ └─ app_helpers.py\
│ │ \
│ ├── templates\
│ │ ├─ evaluate.html\
│ │ ├─ index.html\
│ │ └─ predict.html\
│ │ \
│ └── run.py \
│ \
├── scripts\
│ ├── exp_configs \
│ │ ├─ config_densenet_scratch.json \
│ │ ├─ config_densenet_transfer.json \
│ │ ├─ config_resnet18_scratch.json \
│ │ ├─ config_resnet18_transfer.json \
│ │ ├─ config_resnet50_scratch.json \
│ │ ├─ config_resnet50_transfer.json \
│ │ ├─ config_squeezenet_scratch.json \
│ │ ├─ config_squeezenet_transfer.json \
│ │ ├─ config_vgg_scratch.json \
│ │ └─ config_vgg_transfer.json \
│ │ \
│ ├── config.json >>> Template config file for training \
│ ├── evaluate.py >>> Evaluation script \
│ ├── predict_image.py >>> Script for image class prediction \
│ ├── run.exp.sh >>> Helper shell script to run several trainings/evaluations using configs from exp_configs folder\
│ └── train.py >>> Training script \
│ \
├── source\
│ ├── helper_ml_pipeline.ipynb >>> Helper Jupyter Notebook for development of functions in functions.py \
│ └── functions.py >>> File containing helper functions \
│ \
├── screenshots\
│ ├── class_distribution.png \
│ ├── dog_example.png \
│ ├── app_evaluate.png \
│ ├── app_predict.png \
│ └── app_start.png \
│ \
├── requirements.txt\
└── README.md

## Description of repository and scripts<a name=Chap4></a>

This is a python project that provides 3 main python scripts: 
* `train.py` to train a model on the training dataset split.
* `evaluate.py` to evaluate a trained model.
* `predict_image.py` to predict a class for an image of your choice. 

In addition, a web-app is available that can be used for nicely visualizations of the training and evaluation results, 
as well as for prediction of image classes. Currently, the web-app **cannot** be used for training, therefore a model
must be trained before using the `train.py` script. 

See some screenshots to get an idea of how the web-app looks like.
#### Start screen
![Start](./screenshots/app_start.png?raw=true "Start")
#### Evaluation screen
![Evaluate](./screenshots/app_evaluate.png?raw=true "Evaluate")
#### Prediction screen
![Predict](./screenshots/app_predict.png?raw=true "Predict")

Note that the web-app is designed to work on a local system only, because it requires knowledge about your file system
to avoid uploading models and huge datasets.

Using a configuration file (.json), one can set the model type that should be used for training, as well as many other
parameters to tune the training. Currently available models are listed above in the description.

Here is a list for all parameters that can be set in the configuration file:
* `model`: Name of the model that should be used for training. Must be one of the models in the list above.
* `work_dir`: Path to a working directory. This directory is used to place all values and intermediate data that is
            gained during training. The configuration file used during training is also copied to this folder.
* `train_from_scratch`: If set to `True`, the model is trained with randomly initialized weigths. If `False`, 
                        pretrained weights (trained on [ImageNet](https://image-net.org/)) are downloaded from the
                        pytorch model zoo, and the model is trained later on the dog breed images. This method is known
                        as transfer learning. 
* `train_dataset`: Path to the training dataset folder.
* `val_dataset`: Path to the validation dataset folder.
* `test_dataset`: Path to the test dataset folder. 
* `num_epochs`: Number of epochs the training is running.
* `batch_size`: Batch size that is used during training.
* `optimizer`: Optimization algorithm used for training. Either "sgd" or "adam". 
* `lr`: Learning rate used for the optimization.
* `momentum`: Momentum used for the optimization. Is ignored in case of "adam".
* `weight_decay`: Used weight decay. 
* `lr_scheduler_step_size`: Step size for the learning rate scheduler. This means, if set to $n$, after each $n$-th epoch
                            the learning rate is mulitplied by the factor given in `lr_scheduler_fac`.
* `lr_scheduler_fac`: Factor for learning rate scheduler.
* `use_loss_weights`: If set to `True`, the loss uses weights for each class, trying to correct inbalancing the training dataset. 
* `use_cuda`: If set to `True`, a CUDA-capable GPU device is used for training. 
* `val_epoch`: If set to $n$, after each $n$-th epoch the currently model is evaluated on a subset of the validation dataset. 
               If the model is better then during the evaluation before, it's model state is saved to the working directory. 
* `val_percentage`: Percentage (given in $[0,1]$) of how many samples of the validation set should be used for validation during training.
* `val_top_k`: The evaluation calclated the accuracy, the top-1 error and the top-k error for the $k$ given here. 


## Installation and application instructions <a name=Chap5></a>

In the following, it is explained how to install the required packages and how to run a training, evaluation, and 
prediction on the Dog Breed Dataset. 

1. Install Python 3.7.3 or higher. Furthermore, install the required Python packages using
a virtual environment and the `requirements.txt` file. E.g., for Linux systems run the following lines in the project's root:
    ```console
    python -m venv venv
    source venv/bin/activate
    python -m pip install -r requirements.txt
    ```
    **Note:** When training on a CUDA-capable GPU, additionaly installation steps might be required!
2. Download the Dog Breed Dataset as described [here](https://github.com/udacity/dog-project).
3. Train a model. This can be done using the template configuration file:
    ```console
    python scripts/train.py --config_file scripts/config.json
    ```
    You need to adapt the configuration file before, e.g., adapt paths to your dataset splits. You are also free to 
    use other configuration files. 
4. Evaluate the trained model. For this, eihter call 
    ```console
    python scripts/evaluate.py --config_file <PATH_TO_YOUR_WORKING_DIR>/config.json
    ```
    or use the web-app (see steps 6. below). Please note, that there are additional options that you can set, 
    e.g., you can set the usage of a CUDA-capable device with `--use_cuda`.
5. Use the model to predict the class of an arbitray image. For this, eihter call 
    ```console
    python scripts/predict_image.py <IMAGE_PATH> --config_file <PATH_TO_YOUR_WORKING_DIR>/config.json
    ```
    or use the web-app (see steps 6. below). Please note, that there are additional options that you can set, 
    e.g., you can set the usage of a CUDA-capable device with `--use_cuda`.
6. To evaluate a model or predict the class of an image using the web-app, run
    ```console
    python app/run.py
    ```
    and follow the instructions in the web-app. The web-app is hosted at http://0.0.0.0:3001/ or localhost:3001/


## Dependencies <a name=Chap6></a>

This project uses the following python libraries:
* [argparse](https://docs.python.org/3/library/argparse.html): Parser for command-line options
* [flask](https://flask.palletsprojects.com/en/2.0.x/): Python based web framework
* [numpy](https://numpy.org/): Library for N-dimensional arrays
* [os](https://docs.python.org/3/library/os.html): Miscellaneous operating system interfaces
* [pandas](https://pandas.pydata.org/): Library to handle datasets
* [Pillow](https://pillow.readthedocs.io/en/stable/): Python Image Library
* [plotly](https://plotly.com/): Interactive, open-source, and browser-based graphing library for Python
* [re](https://docs.python.org/3/library/re.html): Library to handle regular expressions
* [scikit-learn](https://scikit-learn.org/stable/): Machine Learning library
* [sys](https://docs.python.org/3/library/sys.html): System specific params and functions
* [torch](https://pytorch.org/docs/stable/torch.html): Package for multi-dimensional tensors and operations over these.
* [torchvision](https://pytorch.org/vision/stable/index.html): Visualization for torch

## Dog Breed Classificaiton -- Results <a name=Chap7></a>

As explained in the description above, a training using either transfer-learning or training from scratch
was performed on all network types using the same training parameters. 
The configuration files for the trainings can be found at 

scripts\
│ └── exp_configs

You can adapt and use the following script, 
```console
scripts/run_exp.sh
```
to train and evaluate all, or just a subset of the experimental configuration files. 

In addition to the accuracy and top-1 error (which are pretty much the same), I also 
evaluated the top-3 error. The motivation behind this is, that there are many dog breeds
that are very hard to distinguish, even for human eyes. Therefore, it makes sense to 
check if a classifier contains the correct class at least in the top-3 predictions.

Here are the results of the evaluations on the test dataset split:

| **Model**                           | **Accuracy** | **Top-1 error** | **Top-k error** |
| ----------------------------------- |-------------:|----------------:|----------------:|
| **densenet (from scratch)**         | 0.1005       | 0.8994          | 0.7796          |
| **densenet (transfer-learning)**    | 0.8842       | 0.1157          | 0.0229          |
| **resnet18 (from scratch)**         | 0.1764       | 0.8235          | 0.6646          |
| **resnet18 (transfer-learning)**    | 0.8507       | 0.1492          | 0.0385          |
| **resnet50 (from scratch)**         | 0.0970       | 0.9029          | 0.7760          |
| **resnet50 (transfer-learning)**    | **0.9065**   | **0.0934**      | **0.0131**      |
| **squeezenet (from scratch)**       | 0.0263       | 0.9736          | 0.9245          |
| **squeezenet (transfer-learning)**  | 0.7441       | 0.2558          | 0.0744          |
| **vgg (from scratch)**              | 0.2143       | 0.7856          | 0.5760          |
| **vgg (transfer-learning)**         | 0.8758       | 0.1441          | 0.0291          |

It is obvious, that training with transfer-learning is much better than training from scratch. 
Comparing the models with each other, **resnet50** is the winner, achieving the best evaluation
results.

Having a look on the training loss and evaluation of the resnet50 training, one can also see
that no overfitting issues seem to appear, so the resulting classifier should be the best choice here.

![Eval](./screenshots/eval.png?raw=true "Eval")
![Loss](./screenshots/loss.png?raw=true "Loss")


## Project conclusion <a name=Chap8></a>

In this project, a Deep Learning classification workflow based on pytorch has been implemented and tested on
the Dog Breed Dataset. Several models has been trained on the dataset using both, transfer-learning and training from
scratch. All the models has been evaluated on the test dataset split to find the best model.

In addition to transfer-learning, loss weights are used to reduce the impact of the slightly inbalanced training dataset. 

Further ideas for improvements are the following:
* Make parameter tuning to get the best model.
* Speed-up the training using parallel processing and/or data caching.
* Add model training to the web-app.

## License <a name=Chap9></a>

The code is free for any usage. For the license of the Dod Breed dataset, please check the license of data provided
[here](https://github.com/udacity/dog-project). 
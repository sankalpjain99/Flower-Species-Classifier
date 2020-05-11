# Flower-Species-Classifier
### This project is part of Udacity's Introduction to Machine Learning with PyTorch Nanodegree


In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application.


## Dataset

The dataset used can be found [here] ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html"). It consists of photos of flowers of 102 different species and our task is to identify them correctly.

The data need to comprised of 3 folders
   - test
   - train
   - validate 

Generally the proportions should be 70% training 10% validate and 20% test.Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image a.jpj and it is a rose it could be in a path like this /test/5/a.jpg and json file would be like this {...5:"rose",...}. 

## How to RUN?

1. Train a new network on a data set with train.py

    - Basic usage: python train.py data_directory
    - Prints out training loss, validation loss, and validation accuracy as the network trains
    - Options:
      - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
      - Choose architecture(densenet121 and vgg16 are available): python train.py data_dir --arch "vgg16"
      - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
      - Use GPU for training: python train.py data_dir --gpu

2. Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

    - Basic usage: python predict.py /path/to/image checkpoint
    - Options:
      - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
      - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
      - Use GPU for inference: python predict.py input checkpoint --gpu
      

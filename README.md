# Endo_recognition Classifier

## Description

## Prerequisites

You will need the following things properly installed on your computer.

* [Docker](https://www.docker.com/)

## Installation

* `git clone `

## Running

##train for best model
nvidia-docker exec endo_rec python3 train.py -m 'skin_rec' --gpus 0,1,2,3,4,5,6,7

##test for best model
nvidia-docker exec endo_rec python3 test.py  -m 'skin_rec' --gpus 0,1,2,3,4,5,6,7

Remember that Docker container has the Python version 3.5.2!

1. Download data
1. Download pretrained .
2. If you are planning to use nvidia-docker, you need to building nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t mm_keras_tf_py3:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/imaterialist -dt --name endo_rec mm_keras_tf_py3:gpu /bin/bash
    ```
3. Download images
    ```
    nvidia-docker exec endo_rec python3 download.py data/test.json data/raw/test
    nvidia-docker exec endo_rec python3 download.py data/train.json data/raw/train
    nvidia-docker exec endo_rec python3 download.py data/validation.json data/raw/validation
    ```
4. Training
    ```bash
    nvidia-docker exec endo_rec python3 train.py [-h]
    ```
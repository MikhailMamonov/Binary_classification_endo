"""
Trains model

Usage: python train.py [-h]
"""
from argparse import ArgumentParser
from multiprocessing import cpu_count
from os import path, environ
import pandas as pd
import numpy as np
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.preprocessing.image import ImageDataGenerator
from utils import (TEST_DATA_PATH,CWD, TRAIN_DATA_PATH, VALIDATION_DATA_PATH,
                   MODELS_PATH, CLASSES, try_makedirs, plot_loss_acc,
                   plot_confusion_matrix)
from sklearn.metrics import confusion_matrix
from models import get_model
from config import config
from keras.utils import multi_gpu_model
from keras.models import load_model

def init_argparse():
    """
    Initializes argparse

    Returns parser
    """
    parser = ArgumentParser(description='Trains toxic comment classifier')
    parser.add_argument(
        '-m',
        '--model',
        nargs='?',
        help='model architecture (vgg16, vgg19, incresnet, incv3, xcept, resnet50, densnet, nasnet)',
        default='vgg16',
        type=str)
    parser.add_argument(
        '--gpus',
        nargs='?',
        help="A list of GPU device numbers ('1', '1,2,5')",
        default=0,
        type=str)
    return parser


def train_and_predict(model_type, gpus):
    """
    Trains model and makes predictions file
    """
    # creating data generators
    print("test path")
    
    print(TEST_DATA_PATH)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_PATH,
        class_mode='binary',        
        **config[model_type]['flow_generator'])

    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_PATH,
        class_mode='binary',
        shuffle=False,
        **config[model_type]['flow_generator'])
    
    print(test_generator.classes)
    
    print('Loading model')
    loading_model = path.join(MODELS_PATH, 'skin_rec_0.3498_0.8740')
    print(loading_model)
    model = load_model(loading_model + '/model.h5')
    



    print('Generating predictions')
    predictions = model.predict_generator(
        test_generator,
        max_queue_size=100,
        use_multiprocessing=True,
        workers=cpu_count())
    pred_classes = np.argmax(predictions, axis=1)
    #pred_classes = model.predict(TEST_DATA_PATH)
    print(pred_classes)
    # Dealing with missing data
    ids = list(map(lambda id: id[5:-4], test_generator.filenames))
    proba = predictions[np.arange(len(predictions)), pred_classes]
    # Generating predictions.csv for Kaggle
    pd.DataFrame({
        'id': ids,
        'predicted': pred_classes,
    }).sort_values(by='id').to_csv(
        path.join(CWD, 'predictions2.csv'), index=False)
    # Generating predictions.csv with some additional data for post-processing
    pd.DataFrame({
        'id': ids,
        'predicted': pred_classes,
        'proba': proba
    }).sort_values(by='id').to_csv(
        path.join(CWD, 'predictions2_extd.csv'), index=False)


def main():
    """
    Main function
    """
    args = init_argparse().parse_args()
    train_and_predict(args.model, args.gpus)


if __name__ == '__main__':
    main()

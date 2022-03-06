import os, shutil
import numpy as np
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
import argparse

def argparser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ['true', 'yes']:
            return True
        elif v.lower() in ['false', 'no']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(description='hyperparameters for training')

    parser.add_argument('--feat.train-tfr',  nargs='+', required=True, dest='train_tfr', help='Directories with feature an labels for training the neural network')
    parser.add_argument('--feat.test-tfr', type=str, default=None, dest='test_tfr', help='Directory with feature an labels for test')
    parser.add_argument('--feat.validation-tfr', type=str, default=None, dest='validation_tfr', help='Directory with feature an labels for validation')
    parser.add_argument('--trainer.continue-learning', type=str2bool, default=False, dest='continue_learning', help='boolean flag, whether flat start or transfer learning')
    parser.add_argument('--trainer.activation-function', type=str, default=None, dest='activation_function', help='an string for activation function')
    parser.add_argument('--trainer.batch-size', type=int, default=128, dest='batch_size', help='an integer for batch size')
    parser.add_argument('--trainer.learning-rate', type=float, default=0.00001, dest='learning_rate', help='an floating point for learning rate')
    parser.add_argument('--trainer.epoch', type=int, default=None, dest='epoch', help='an integer for epochs')
    parser.add_argument('--dir', type=str, default=None, dest='dir', help='Directory to store the models and all other files.' )
    
    args = parser.parse_args()

    if args.epoch <=0:
        raise Exception("--trainer.epoch should be non-negative")
    if args.dir == '':
        raise Exception("--dir should be not empty")
    if len(args.train_tfr) == 0 or  args.test_tfr == '' or  args.validation_tfr == '':
        raise Exception("Directory of TFrecord should be not empty")
    
    return args.train_tfr, args.test_tfr, args.validation_tfr, args.continue_learning, args.activation_function, args.batch_size, args.learning_rate, args.epoch, args.dir

def dir_setting(dir_name, CONTINUE_LEARNING): 
    cp_path = os.path.join(os.getcwd(), dir_name)   # cp_path = check point path
    confusion_path = os.path.join(cp_path, 'confusion_matrix')
    model_path = os.path.join(cp_path, 'model')

    if CONTINUE_LEARNING == False and os.path.isdir(cp_path):
        shutil.rmtree(cp_path)
    
    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True)   # cp_path가 없으면 만들고, 없으면 그냥 무시
        os.makedirs(confusion_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

    path_dict = {'cp_path': cp_path,
                'confusion_path': confusion_path,
                'model_path':model_path}
    return path_dict

def get_classification_metrics():
    train_loss = Mean()
    train_acc = CategoricalAccuracy()

    validation_loss = Mean()
    validation_acc = CategoricalAccuracy()

    test_loss = Mean()
    test_acc = CategoricalAccuracy()

    metric_objects = dict()
    metric_objects['train_loss'] = train_loss
    metric_objects['train_acc'] = train_acc
    metric_objects['validation_loss'] = validation_loss
    metric_objects['validation_acc'] = validation_acc
    metric_objects['test_loss'] = test_loss
    metric_objects['test_acc'] = test_acc

    return metric_objects

def continue_setting(CONTINUE_LEARNING, path_dict, model=None):
    if CONTINUE_LEARNING == True and len(os.listdir(path_dict['model_path'])) == 0:
        CONTINUE_LEARNING = False
        print(colored('CONTINUE_LEARNING flag has been converted to FALSE.', 'cyan'))

    if CONTINUE_LEARNING == True:
        epoch_list = os.listdir(path_dict['model_path'])
        epoch_list = sorted([int(epoch.split('_')[1]) for epoch in epoch_list])
        last_epoch = epoch_list[-1]
        model_path = path_dict['model_path'] + '/epoch_' + str(last_epoch)
        model = tf.keras.models.load_model(model_path)

        losses_accs_path = path_dict['cp_path']
        losses_accs_np = np.load(losses_accs_path + '/losses_accs.npz')
        losses_accs = dict()
        for k, v in losses_accs_np.items():
            losses_accs[k] = list(v)
        
        start_epoch = last_epoch + 1

    else:
        model = model
        start_epoch = 0
        losses_accs = {'train_losses':[], 'train_accs':[],
                        'validation_losses':[], 'validation_accs':[]}
                        
    return model, losses_accs, start_epoch

import os

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from utils.learning_env_setting import dir_setting, continue_setting, get_classification_metrics, argparser
from utils.basic_utils import resetter, training_reporter
from utils.train_validation_test import train, validation, test
from utils.cp_utils import save_metrics_model, metric_visualizer
from models.basic_models import DenseLayer1

os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='ture'

############################################### Learning Setting  ##################################################
continue_learning = False  # False : 처음 돌리거나 지금까지 돌린게 있으면 모두 지우고 돌릴때 설정!
save_period=2

train_tfr, test_tfr, validation_tfr, continue_learning, activation_function, batch_size, learning_rate, epochs, exp_name = argparser()

model = DenseLayer1(activation=activation_function)
optimizer = SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
####################################################################################################################

loss_object = CategoricalCrossentropy()
path_dict = dir_setting(exp_name,continue_learning)
model, losses_accs, start_epoch = continue_setting(continue_learning, path_dict, model=model)

def deserialize_example(serialized_string):
    feature_description = {
        'feature' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_string, feature_description)
    feature = tf.io.decode_raw(example["feature"], tf.float32)
    label = tf.io.decode_raw(example["label"], tf.float32)

    return feature, label

train_ds = tf.data.TFRecordDataset(train_tfr).map(deserialize_example).shuffle(1000).batch(batch_size).prefetch(batch_size)
validation_ds = tf.data.TFRecordDataset(validation_tfr).map(deserialize_example).shuffle(1000).batch(batch_size).prefetch(batch_size)
test_ds = tf.data.TFRecordDataset(test_tfr).map(deserialize_example).shuffle(1000).batch(batch_size).prefetch(batch_size)

metric_objects = get_classification_metrics()

if not model.built:  # model이 build 되지 않았으면....
    model.build(input_shape=(None, 3500))   # None --> batch size를 정해주지 않았기 때문에 None으로 설정 

for epoch in range(start_epoch, epochs):
    train(train_ds, model, loss_object, optimizer, metric_objects)
    validation(validation_ds, model, loss_object, metric_objects)
    training_reporter(epoch, losses_accs, metric_objects)
    save_metrics_model(epoch, model, losses_accs, path_dict, save_period)

    metric_visualizer(losses_accs, path_dict['cp_path'])
    resetter(metric_objects)

test(test_ds, model, loss_object, metric_objects, path_dict)

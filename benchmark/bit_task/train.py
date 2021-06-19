# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# coding: utf-8

from functools import partial
import time
import os
import argparse

from . import bit_hyperrule
from . import input_pipeline
from .callbacks import LRSchedule, TimeLimit, myModelCheckpoint
from .model_builder import MyHyperModel, build_from_ckpt
from . import config # self defined

import kerastuner as kt
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

class MyTuner(kt.Tuner):


    def run_trial(self,
                  trial,
                  train_data,
                  valid_data,
                  test_data,
                  num_examples,
                  search_space) :

        # 1. get hp for this trial
        hp = trial.hyperparameters

        batch_size = hp.Choice('batch_size', search_space['hyperparameters']['batch_size'])
        epochs = hp.Choice('epochs', search_space['hyperparameters']['epochs'])
        lr = config.get_kt_lr(hp, search_space['hyperparameters']['lr'])

        self._display.on_trial_begin(trial) # display hyperparameters

        # 2. split batches, prepare datasets
        train_data = train_data.unbatch().batch(batch_size, drop_remainder=True).prefetch(1)
        valid_data = valid_data.unbatch().batch(1, drop_remainder=True).prefetch(1)
        print(train_data)

        # 3. build model
        model = self.hypermodel.build(hp)

        # 4. call back implementation
        TimeLimitCallBack = TimeLimit(search_space['tune_kwargs']['start_time'],
                                      search_space['tune_kwargs']['time_limit'],
                                      test_data, config.LOGGER)
        LRScheduleCallBack = LRSchedule(lr, num_examples)
        EarlyStopCallback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=3, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True)
        ModelCheckpointCallback= myModelCheckpoint(
                filepath=config.CKPT_PATH,
                save_best_only=True,
                save_weights_only=True,
                monitor='val_accuracy',
                modemode='max', verbose=1,
                model_name = hp['model'])
        best_trials = self.oracle.get_best_trials()
        if len(best_trials) > 0 :
            ModelCheckpointCallback.best = best_trials[0].score # send previous best score into current callback

        # 5. start training
        history = model.fit(train_data,
                            steps_per_epoch=100,
                            epochs=epochs,
                            verbose = 2,
                            validation_data=valid_data,
                            callbacks=[LRScheduleCallBack,
                                       ModelCheckpointCallback,
                                       EarlyStopCallback,
                                       TimeLimitCallBack],)

        # 6. update trial, must do
        self.oracle.update_trial(trial.trial_id, {'val_accuracy':max(history.history['val_accuracy'])})


    def on_trial_begin(self, trial) :
        if self.logger :
            self.logger.register_trial(trial.trial_id, trial.get_state())
        # self._display.on_trial_begin(trial) # removed since it doesn't show runtime hp






def argparser():
    parser = argparse.ArgumentParser(description="BiT with HPO.")
    parser.add_argument("--name", required=True,
                          help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument("--output_path", default='/home/yiran.wu/wyr/output/v3/',
                          help="Where to log training info (small).")
    parser.add_argument("--pretrained_path", default='/home/yiran.wu/Bit_Models/',
                          help="Where to search for pretrained BiT models.")
    parser.add_argument("--data_path", required=True, help="Set your dataset directory")
    parser.add_argument("--overwrite", type=bool, default=True,
                          help="Overwrite the trials of the folder with name passed by --name.")
    return parser.parse_args()



'''
def config_path(args):
    config.Name = args.name
    config.DATA_PATH = args.data_path
    config.PRETRAINED_PATH = args.pretrained_path
    config.OUTPUT_PATH = os.path.join(args.output_path, args.name)
    config.CKPT_PATH = os.path.join(config.OUTPUT_PATH, 'best_so_far/')
    config.TRIALS_PATH = os.path.join(config.OUTPUT_PATH, 'trials')
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    if not os.path.exists(config.CKPT_PATH):
        os.makedirs(config.CKPT_PATH)
'''

def configure_opts_path(opts):
    config.Name = opts.dataset
    config.DATA_PATH = opts.data_path
    config.PRETRAINED_PATH = '/home/yiran.wu/Bit_Models/'
    config.OUTPUT_PATH = os.path.join(opts.output_path, 'bit_task', config.Name)
    config.CKPT_PATH = os.path.join(config.OUTPUT_PATH, 'best_so_far/')
    config.TRIALS_PATH = os.path.join(config.OUTPUT_PATH, 'trials/')
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    if not os.path.exists(config.CKPT_PATH):
        os.makedirs(config.CKPT_PATH)




def bit_start(opts):
    # 1. get args, update config global vars
    configure_opts_path(opts)
    config.SEARCH_SPACE['tune_kwargs']['start_time'] = time.time()  # set search config
    config.LOGGER = config.setup_logger(args=opts, logdir= config.OUTPUT_PATH) # set up logger
    config.LOGGER.critical(f'Available devices: {tf.config.list_physical_devices()}')


    # 2. set up Distribute training
    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(num_devices))


    # 3. dataset input pipeline
    (train_data, valid_data, test_data, num_examples, num_classes) = \
        input_pipeline.get_data(dataset=config.DATA_PATH, train_split=config.TRAIN_SPLIT)
    config.NUMCLASSES = num_classes

    # 4. build hypermodel
    hyperparameters = config.SEARCH_SPACE['hyperparameters']
    tune_kwargs= config.SEARCH_SPACE['tune_kwargs']
    hypermodel = MyHyperModel(hyperparameters=hyperparameters,
                              num_classes=num_classes,
                              strategy=strategy,
                              pretrained_dir=config.PRETRAINED_PATH)


    # 5. starch searching
    config.LOGGER.critical("Hypermodel built. Start Searching.")
    oracle = config.get_kt_strategy(tune_kwargs['search_strategy'])
    oracle = oracle(objective=kt.Objective('val_accuracy', 'max'),
                    max_trials=tune_kwargs['num_trials'])

    tuner = MyTuner(oracle=oracle,
                    hypermodel = hypermodel,
                    directory=config.TRIALS_PATH,
                    project_name=config.Name,
                    overwrite = True)

    tuner.search( train_data=train_data,
                  valid_data=valid_data,
                  test_data=test_data,
                  num_examples=num_examples,
                  search_space=config.SEARCH_SPACE)

    best_hps = tuner.get_best_hyperparameters()[0]
    config.LOGGER.critical(f"Best HP values: {best_hps.values}")

    # 6. get test accuracy for best model
    best_model = build_from_ckpt(config.CKPT_PATH, config.NUMCLASSES, config.BESTMODELFILE)
    loss, acc = best_model.evaluate(test_data, verbose=2)
    config.LOGGER.critical("Restored best model, Test accuracy: {:5.2f}%, loss: {:5.2f}".format(100 * acc, loss))
    config.LOGGER.critical(f'All files saved to {config.OUTPUT_PATH}')


      # vim /home/yiran.wu/.local/lib/python3.7/site-packages/kerastuner/engine/tuner.py : added in line 317: return ...
      # best_model = tuner.get_best_models()[0]

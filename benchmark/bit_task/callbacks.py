import os
import time
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()


import config
import bit_hyperrule
from model_builder import build_from_ckpt

class LRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, base_lr, num_samples, step = 0):
        self.step = step
        self.base_lr = base_lr
        self.num_samples = num_samples

    def on_train_batch_begin(self, batch, logs=None):
        lr = bit_hyperrule.get_lr(self.step, self.num_samples, self.base_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.step += 1


class TimeLimit(tf.keras.callbacks.Callback):
    def __init__(self, start_time, max_time):
        self.start_time = start_time
        self.max_time = max_time

    def on_epoch_end(self, epoch, logs=None):
         time_elapsed = time.time() - self.start_time
         if(time_elapsed > self.max_time):
            self.model.stop_training = True



# class myModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
#     def __init__(self,
#                filepath,
#                monitor='val_loss',
#                verbose=0,
#                save_best_only=False,
#                save_weights_only=False,
#                mode='auto',
#                save_freq='epoch',
#                options=None,
#                model_name=None,
#                **kwargs):
#         super(myModelCheckpoint, self).__init__(
#                filepath,
#                monitor=monitor,
#                verbose=verbose,
#                save_best_only=save_best_only,
#                save_weights_only=save_weights_only,
#                mode=mode,
#                save_freq=save_freq,
#                options=options,
#                **kwargs)
#         self.model_name = model_name
#
#     def on_epoch_end(self, epoch, logs=None) :
#         current = logs.get(self.monitor)
#         if self.monitor_op(current, self.best):
#             tf.io.gfile.makedirs(self.filepath)
#             with open(os.path.join(self.filepath, config.BESTMODELFILE), 'w') as f:
#                 f.write(self.model_name)
#         super(myModelCheckpoint, self).on_epoch_end(epoch, logs)





import os
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from kerastuner import HyperModel
from . import models

def build_from_ckpt(ckpt_path, num_classes, bestmodelfile):
    with open(os.path.join(ckpt_path, bestmodelfile)) as f:
        model_name = f.readline()

    num_out = 1000 if model_name.split('-')[1] == 'S' else 21843
    model = models.ResnetV2(
        num_units=models.NUM_UNITS[model_name],
        num_outputs=num_out,
        filters_factor=int(model_name[-1]) * 4,
        name="resnet",
        trainable=True,
        dtype=tf.float32)
    model.build((None, None, None, 3))

    model._head = tf.keras.layers.Dense(
        units=num_classes,
        use_bias=True,
        kernel_initializer="zeros",
        trainable=True,
        name="head/dense")

    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    model.load_weights(ckpt_path).expect_partial()
    return model



class MyHyperModel(HyperModel) :
    def __init__(self,
                 hyperparameters,
                 num_classes,
                 strategy,
                 pretrained_dir):

        self.num_classes = num_classes
        self.hyperparameters = hyperparameters
        self.strategy = strategy # distribute strategy
        self.pretrained_dir = pretrained_dir

    def build(self, hp) :
        with self.strategy.scope():
            model_name = hp.Choice('model', self.hyperparameters['model'])

            def get_model_file(model, bit_pretrained_dir) :
                tf.io.gfile.makedirs(bit_pretrained_dir)
                bit_model_file = os.path.join(bit_pretrained_dir, f'{model}.h5')
                if not tf.io.gfile.exists(bit_model_file) :
                    print(f"Cannot find {bit_model_file}. Please prepare it before training.")
                    exit(1)
                return bit_model_file

            bit_model_file = get_model_file(model_name, self.pretrained_dir)

            num_out = 1000 if model_name.split('-')[1] == 'S' else 21843
            model = models.ResnetV2(
                num_units=models.NUM_UNITS[model_name],
                num_outputs=num_out,
                filters_factor=int(model_name[-1]) * 4,
                name="resnet",
                trainable=True,
                dtype=tf.float32)

            model.build((None, None, None, 3))
            model.load_weights(bit_model_file)
            print(f'Weights loaded into model {model_name}!')

            model._head = tf.keras.layers.Dense(
                units=self.num_classes,
                use_bias=True,
                kernel_initializer="zeros",
                trainable=True,
                name="head/dense")

            optimizer = tf.keras.optimizers.SGD(momentum=0.9)
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        return model


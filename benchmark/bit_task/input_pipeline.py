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

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

from . import bit_hyperrule

# A workaround to avoid crash because tfds may open too many files.
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000

# vim /home/yiran.wu/.local/lib/python3.7/site-packages/tensorflow_datasets/core/dataset_info.py :
#                                                       added in lin 449 : return

def get_data(dataset, train_split):
    
    resize_size, crop_size = bit_hyperrule.get_resolution_from_dataset(dataset)
  
    # build from folder
    data_builder = tfds.folder_dataset.ImageFolder(dataset)

    # get numbers
    num_classes = data_builder.info.features['label'].num_classes
    num_train = data_builder.info.splits['train'].num_examples
    num_test =  data_builder.info.splits['test'].num_examples
    num_valid = data_builder.info.splits['val'].num_examples
    print(num_valid)
    # to dataset
    train_data = data_builder.as_dataset(split='train', decoders={'image': tfds.decode.SkipDecoding()})
    test_data  = data_builder.as_dataset(split='test', decoders={'image' : tfds.decode.SkipDecoding()})
    valid_data = data_builder.as_dataset(split='val', decoders={'image' : tfds.decode.SkipDecoding()})

    decoder = data_builder.info.features['image'].decode_example
    mixup_alpha=bit_hyperrule.get_mixup(num_train)


    # get returns
    train_data = data_aug(data=train_data,
                          mode='train',
                          num_examples=num_train,
                          decoder=decoder,
                          num_classes=num_classes,
                          resize_size=resize_size,
                          crop_size=crop_size,
                          mixup_alpha=mixup_alpha)

    valid_data = data_aug(data=valid_data,
                          mode='valid',
                          num_examples=num_valid,
                          decoder=decoder,
                          num_classes=num_classes,
                          resize_size=resize_size,
                          crop_size=crop_size,
                          mixup_alpha=mixup_alpha)

    test_data = data_aug(data=test_data,
                          mode='test',
                          num_examples=num_test,
                          decoder=decoder,
                          num_classes=num_classes,
                          resize_size=resize_size,
                          crop_size=crop_size,
                          mixup_alpha=mixup_alpha)

    return train_data, valid_data, test_data, num_train, num_classes
    
    
    
# shadow function of get_data
def data_aug(data,
             mode,
             num_examples,
             decoder,
             num_classes,
             resize_size,
             crop_size,
             mixup_alpha):

    def _pp(data):
        im = decoder(data['image'])
        if mode == 'eee':
            im = tf.image.resize(im, [resize_size, resize_size])
            im = tf.image.random_crop(im, [crop_size, crop_size, 3])
            im = tf.image.flip_left_right(im)
        else:
            # usage of crop_size here is intentional
            im = tf.image.resize(im, [crop_size, crop_size])
        im = (im - 127.5) / 127.5
        label = tf.one_hot(data['label'], num_classes)
        return {'image': im, 'label': label}

    def _mixup(data):
        beta_dist = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
        beta = tf.cast(beta_dist.sample([]), tf.float32)
        data['image'] = (beta * data['image'] +
                         (1 - beta) * tf.reverse(data['image'], axis=[0]))
        data['label'] = (beta * data['label'] +
                         (1 - beta) * tf.reverse(data['label'], axis=[0]))
        return data


    def reshape_for_keras(features, crop_size):
        features["image"] = tf.reshape(features["image"], (1, crop_size, crop_size, 3))
        features["label"] = tf.reshape(features["label"], (1, -1))
        return (features["image"], features["label"])


    data = data.cache()
    if mode == 'train':
        data = data.repeat(None).shuffle(min(num_examples, MAX_IN_MEMORY))
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)
    data = data.batch(1)
    # if mixup_alpha is not None and mixup_alpha > 0.0 and mode == 'train':
    #     data = data.map(_mixup, tf.data.experimental.AUTOTUNE)
    data = data.map(lambda x: reshape_for_keras(x, crop_size=crop_size))

    return data


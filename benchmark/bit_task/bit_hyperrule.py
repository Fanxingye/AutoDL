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
import math
def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96*96 else (512, 480)


known_dataset_sizes = {
    'cifar10': (32, 32),
    'cifar100': (32, 32),
    'oxford_iiit_pet': (224, 224),
    'oxford_flowers102': (224, 224),
    'imagenet2012': (224, 224),
    '/data/AutoML_compete/datasciencebowl/' : (105, 100),
    '/data/AutoML_compete/leaf-classification/split/' : (90, 90),
    '/data/AutoML_compete/oxford-102-flower-pytorch/flower_data/': (500, 500),
    '/data/AutoML_compete/cassava-leaf-diease/': (500, 500),
    '/data/AutoML_compete/dog-breed-identification/format/': (500, 500),
    '/data/AutoML_compete/the-nature-conservancy-fisheries-monitoring/train/': (500, 500),
}


def get_resolution_from_dataset(dataset):
    # if dataset not in known_dataset_sizes:
    #   raise ValueError(f"Unsupported dataset {dataset}. Add your own here :)")
    return (256, 224)


def get_mixup(dataset_size):
    return 0.0 if dataset_size < 20_000 else 0.1


def get_schedule(dataset_size):
    if dataset_size < 20_000:
      return [100, 200, 300, 400, 500]
    elif dataset_size < 500_000:
      return [500, 3000, 6000, 9000, 10_000]
    else:
      return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
    """Returns learning-rate for `step` or None at the end."""
    supports = get_schedule(dataset_size)
    # Linear warmup
    if step < supports[0]:
        return base_lr * step / supports[0]
    # End of training
    elif (supports[0] == 100):
        return base_lr / math.pow(2, step//100 - 1)
    else:
        return base_lr / math.pow(3, step//3000)

    # # Staircase decays by factor of 10
    # else:
    #   for s in supports[1:]:
    #     if s < step:
    #       base_lr /= 10
    #   return base_lr

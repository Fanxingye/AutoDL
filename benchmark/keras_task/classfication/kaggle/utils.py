import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
import pandas as pd

def image_dataset_from_directory(directory,
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 interpolation='bilinear'):
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            'Received: %s' % (color_mode,))
    interpolation = image_preprocessing.get_interpolation(interpolation)
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    dataset = path_ds.map(
        lambda x: path_to_image(x, image_size, num_channels, interpolation))
    dataset = dataset.batch(batch_size)
    dataset.class_names = class_names
    dataset.file_paths = image_paths
    return dataset


def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def generate_prob_csv(test_result, dataset_path, input_csv, output_csv, ):
    image_paths = [os.path.splitext(filename)[0] for filename in os.listdir(dataset_path)]
    csv_path = dataset_path.replace('test', input_csv)
    df = pd.read_csv(csv_path, dtype={'id': str})
    row_index_group = []
    for i in image_paths:
        row_index = df[df['id'] == str(i)].index.tolist()
        if not len(row_index) == 0:
            row_index_group.append(row_index[0])

    df.loc[row_index_group, 1:] = test_result.tolist()
    df.to_csv(output_csv, index=False)

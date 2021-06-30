import os
import math
import shutil
import argparse
import numpy as np
import random

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
        '.tiff', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for different kaggle competitions.')
    parser.add_argument('--data-dir', type=str, default='',
                        help='training and validation pictures to use.')
    parser.add_argument('--dataset', type=str, default='train',
                        help='the kaggle competition')
    parser.add_argument('--sampling_strategy', type=str, default='balanced', choices=['balanced', 'random'],
                        help='Sampling strategy, balanced or random')
    parser.add_argument('--split_test', type=bool, default=False, 
                        help='Whether to split test dataset')
    opt = parser.parse_args()
    return opt


def has_file_allowed_extension(filename, extensions=IMG_EXTENSIONS):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def balanced_split(all_data_list, val_ratio=0.1, test_ratio=0.1):
    assert 0 <= val_ratio < 1.0
    assert 0 <= test_ratio < 1.0
    assert 0 < val_ratio + test_ratio < 1.0
    random.shuffle(all_data_list)

    val_nums = math.ceil(len(all_data_list) * val_ratio)
    test_nums = math.ceil(len(all_data_list) * test_ratio)

    val = all_data_list[:val_nums]
    test = all_data_list[val_nums:(val_nums + test_nums)]
    train = all_data_list[(val_nums + test_nums):]
    return train, val, test


def random_split(all_data_list, val_ratio=0.1, test_ratio=0.1):
    assert 0 <= val_ratio < 1.0
    assert 0 <= test_ratio < 1.0
    assert 0 < val_ratio + test_ratio < 1.0

    mask = np.random.rand(len(all_data_list))
    test_mask = mask < test_ratio
    val_mask = (test_ratio < mask) & (mask < test_ratio + val_ratio)
    train_mask = mask > (test_ratio + val_ratio)

    all_data_list = np.array(all_data_list)
    train = all_data_list[train_mask]
    test = all_data_list[test_mask]
    val = all_data_list[val_mask]
    return train, val, test


def copy_images(root_dir, dest_dir, classes, img_list):
    for img_name in img_list:
        img_path = os.path.join(root_dir, classes, img_name)
        isExists = os.path.exists(img_path)
        if (isExists):
            new_path = os.path.join(dest_dir, classes, img_name)
            shutil.copyfile(img_path, new_path)
        else:
            print(str(img_path) + " does not exist.")
    print("%s has been moved to %s" % (classes, dest_dir))


def move_images(root_dir, dest_dir, classes, img_list):
    for img_name in img_list:
        img_path = os.path.join(root_dir, classes, img_name)
        isExists = os.path.exists(img_path)
        if (isExists):
            new_path = os.path.join(dest_dir, classes, img_name)
            shutil.move(img_path, new_path)
        else:
            print(str(img_path) + " does not exist.")
    print("%s has been moved to %s" % (classes, dest_dir))


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("%s does not exist, will be created." % dir)
    else:
        print("%s exists, will be deleted and rebuilt." % dir)
        shutil.rmtree(dir)
        os.makedirs(dir)


def split_train_file():
    opt = parse_args()
    train_path = "split/train/"
    val_path = "split/val/"
    val_ratio = 0.1

    train_path = os.path.join(opt.data_dir, train_path)
    val_path = os.path.join(opt.data_dir, val_path)

    for img_cls in os.listdir(train_path):
        img_cls_dir = os.path.join(train_path, img_cls)
        if os.path.isdir(img_cls_dir):
            img_cls_list = os.listdir(img_cls_dir)
            img_cls_list = [name for name in img_cls_list if has_file_allowed_extension(name)]
            if len(img_cls_list) > 0:
                mkdir(val_path + img_cls)
                if opt.sampling_strategy == "random":
                    train_list, val_list = random_split(img_cls_list, test_size=val_ratio)
                elif opt.sampling_strategy == "balanced":
                    train_list, val_list = balanced_split(img_cls_list, test_size=val_ratio)

            move_images(train_path, val_path, img_cls, val_list)

    print("All images have been processed.")


def main():
    opt = parse_args()
    train_path = "split/train/"
    val_path = "split/val/"
    test_path = "split/test/"
    val_ratio = 0.1
    test_ratio = 0.1

    pic_path = os.path.join(opt.data_dir, opt.dataset)
    train_path = os.path.join(opt.data_dir, train_path)
    val_path = os.path.join(opt.data_dir, val_path)
    test_path = os.path.join(opt.data_dir, test_path)

    for img_cls in os.listdir(pic_path):
        img_cls_dir = os.path.join(pic_path, img_cls)
        if os.path.isdir(img_cls_dir):
            img_cls_list = os.listdir(img_cls_dir)
            img_cls_list = [name for name in img_cls_list if has_file_allowed_extension(name)]
            if len(img_cls_list) > 0:
                mkdir(train_path + img_cls)
                mkdir(val_path + img_cls)
                mkdir(test_path + img_cls)
                if opt.sampling_strategy == "random":
                    train_list, val_list, test_list = random_split(img_cls_list, val_ratio=val_ratio, test_ratio=test_ratio)
                elif opt.sampling_strategy == "balanced":
                    train_list, val_list, test_list = balanced_split(img_cls_list, val_ratio=val_ratio, test_ratio=test_ratio)

            copy_images(pic_path, train_path, img_cls, train_list)
            copy_images(pic_path, val_path, img_cls, val_list)
            copy_images(pic_path, test_path, img_cls, test_list)
        
    print("All images have been processed.")


if __name__ == '__main__':
    main()

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
# from constant import  Constant

from meta_feature_utils import sample_num_strategy # sample strategy
import random
import re

import easyocr  # pip install easyocr
from mtcnn import MTCNN  # pip install mtcnn


from PIL import Image

from numba import cuda


class EngineerFeatureData(object):
    def __init__(self, dict):
        self.name = None
        self.class_count = None
        self.image_count = None
        self.color_mode = None

        # image per class
        self.im_per_class_mean = None
        self.im_per_class_median = None
        self.im_per_class_mode = None
        self.im_per_class_min = None
        self.im_per_class_max = None
        self.im_per_class_range = None
        self.im_per_class_std = None
        self.im_per_class_skew = None
        self.im_per_class_kurt = None

        # image height
        self.height_mean = None
        self.height_median = None
        self.height_mode = None
        self.height_min = None
        self.height_max = None
        self.height_range = None
        self.height_std = None
        self.height_skew = None
        self.height_kurt = None

        # image width
        self.width_mean = None
        self.width_median = None
        self.width_mode = None
        self.width_min = None
        self.width_max = None
        self.width_range = None
        self.width_std = None
        self.width_skew = None
        self.width_kurt = None

        # image area
        self.area_mean = None
        self.area_median = None
        self.area_mode = None
        self.area_min = None
        self.area_max = None
        self.area_range = None
        self.area_std = None
        self.area_skew = None
        self.area_kurt = None

        self.__dict__.update(dict)

import multiprocessing

class EngineerFeature:
    #Constant.ENGINEER_FEATURES_CSV
    def __init__(self, task_config, csv_path='', save_to_file = False):
        ''' Calculate engineered meta features to a dataset, such as num of classes, total count of images

        Args:
            task_config: configs containing job info
            csv_path: path to engineerFeatures feature file
            save_to_file: whether save current data to file, default is False

        Params:
            data_name[str]: name of the dataset
            data_path[str]: path to the dataset
            csv_path[str]: path to the csv file that contains info about previous datasets

            df[pd.DataFrame]: data loaded from csv_path
            entry[np.ndarray]: engineered meta features of current dataset
        '''
        self._contain_chars = False
        self._contain_faces = False
        self._contain_poses = False
        self._is_xray = False

        self.data_name = task_config["data_name"]
        self.data_path = task_config["data_path"]
        self.csv_path = csv_path

        self.df = self._load_csv()
        self.entry = self._generate_feature(save_to_file)

        self.contains = self._judge_special_cases(self.data_path)



    def get_engineered_feature(self) -> EngineerFeatureData:
        ''' Wrap entry to current entry in SimpleNamespace and return

        Returns:
            arg: a SimpleNamespace containing info regarding the dataset.
                Ex: arg.name, arg.im_per_class_median
        '''
        dict = {i : j for i,j in zip(self.df.index, self.entry)}
        dict['name'] = self.data_name
        arg  = EngineerFeatureData(dict)
        return arg

    def contain_chars(self):
        return self._contain_chars

    def contain_faces(self):
        return self._contain_faces

    def contain_poses(self):
        return self._contain_poses

    def is_xray(self):
        return self._is_xray


    def _remove_special_chars(self, input) :
        input = re.sub('[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", input)
        return re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", input)

    def _init_keypoint_detection_predictor(self):
        # python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg

        cfg = get_cfg()  # get a fresh new config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        return  predictor

    def _data_has_char(self, images:list, total_sample) -> bool:
        chars = 0
        reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory

        for im in images:
            res = reader.readtext(im)
            invalid = 0
            for i in res :
                if (self._remove_special_chars(i[1]) == "") :
                    invalid += 1
            if len(res) - invalid > 0:
                chars += 1

        # set threshold
        if chars / total_sample > 0.9:
            self._contain_chars = True
            return True
        return False

    def _data_has_face(self, images:list, total_sample) -> bool:
        faces = 0
        detector = MTCNN()
        for im in images:
            im = np.array(Image.open(im).convert('RGB')).astype(np.float32)
            res = detector.detect_faces(im)

            largest = 0
            for face in res :
                curr = face['box'][0] * face['box'][0]
                largest = curr if curr > largest else largest

            if(largest / 50176 > 0.35):
                faces +=1

        if faces / total_sample > 0.9:
            self._contain_faces = True
            return True
        return False


    def _data_has_pose(self, images:list, total_sample) -> bool:
        poses = 0
        predictor = self._init_keypoint_detection_predictor()

        for im in images:

            im = np.array(Image.open(im).convert('RGB')).astype(np.float32)
            out = predictor(im)

            if len(out['instances'].get_fields()['pred_boxes'].tensor) > 0:
                poses += 1

        if poses/total_sample > 0.9:
            self._contain_poses = True
            return True
        return False


    def _judge_special_cases(self, ddir: str) -> None:
        ''' Get one vector of feature to one dataset

        Args:
            ddir: path to the dataset

        Returns:
            entry: feature vector of one dataset
        '''
        print('Start judging dataset special cases.')
        imPerClass = [len(os.listdir(os.path.join(ddir, i))) for i in os.listdir(ddir)]
        mean = int(np.mean(imPerClass))

        total_sample = 0

        images = []
        for j, c in enumerate(os.listdir(ddir)) :

            im_path = os.path.join(ddir, c)  # path to current class folder
            im_files = os.listdir(im_path)  # image names in the class folder
            class_num = len(im_files)

            sample_num = sample_num_strategy(mean, class_num)
            total_sample += sample_num
            index = random.sample(range(class_num), sample_num)

            for i in index :
                im = os.path.join(im_path, im_files[i])
                images.append(im)

        # multiprocessing.Process(target=self._data_has_face(images, total_sample), )
        if self._data_has_pose(images, total_sample):
            return

        if self._data_has_char(images, total_sample):
            return

        device = cuda.get_current_device()
        device.reset()

        if self._data_has_face(images, total_sample):
            return

        device = cuda.get_current_device()
        device.reset()





    def _generate_feature(self, save_to_file:bool) -> np.ndarray:
        ''' to generate feature
        Used Params:
            self.data_name,
            self.data_path

        Args:
            save_to_file: whether save to file

        Returns:
            entry: entry to current dataset
        '''
        if self.data_name in self.df.columns:
            print(f'{self.data_name} already in csv file so stored features will be loaded. '
                  f'Please use another name if you entered a new dataset.')
            return np.array(self.df[self.data_name])

        entry = self._get_data_features(self.data_path, self.data_name)

        if save_to_file:
            self.df[self.data_name] = entry[1:]
            self.df.to_csv(self.csv_path, header=True, index=True)
        return entry


    def _load_csv(self) -> pd.DataFrame:
        '''

        Args:
            csv_path: path to the csv file

        Returns:
            df: dataframe loaded from the csv file
        '''
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f'Cannot find csv file {self.csv_path}')
        df = pd.read_csv(self.csv_path, index_col=0, dtype='str')

        # convert string to float
        for i in df.index :
            if i == 'color_mode' :
                continue
            df.loc[i] = df.loc[i].astype('float32')

        return df


    def _get_data_features(self, ddir: str, name: str) -> np.ndarray :
        ''' Calculate all the features to the one dataset

        Args:
            ddir: path to the dataset train folder
            name: name of the dataset

        Returns:
            entry: one entry of the engineered features of the dataset
        '''
        imPerClass = [len(os.listdir(os.path.join(ddir, i))) for i in os.listdir(ddir)]
        imPerClass = np.asarray(imPerClass)

        num_classes = len(os.listdir(ddir))
        num_images = np.sum(imPerClass)

        heights = []
        widths = []
        areas = []
        cmode = None

        for c in os.listdir(ddir) :
            for i in os.listdir(os.path.join(ddir, c)) :
                im = Image.open(os.path.join(ddir, c, i))
                size = im.size
                heights.append(size[0])
                widths.append(size[1])
                areas.append(size[0] * size[1])
                cmode = im.mode

        ipc = self._get_list_distribution(imPerClass)
        imh = self._get_list_distribution(np.asarray(heights))
        imw = self._get_list_distribution(np.asarray(widths))
        ima = self._get_list_distribution(np.asarray(areas))
        general = np.asarray([name, num_classes, num_images, cmode], dtype=object)
        entry = np.concatenate((general, ipc, imh, imw, ima))
        return entry

    def _get_list_distribution(self, data: np.ndarray) -> np.ndarray :
        ''' Calculate the statistical info of a list.

        Args:
            data: 1d np array

        Returns:
            out:  the following statistics of input data
                   [ mean, median, mode, min, max,
                    range,std, skewness, kurtosis]
        '''
        out = np.array([np.mean(data),
                        np.median(data),
                        mode(data)[0][0],
                        np.min(data),
                        np.max(data),
                        np.max(data) - np.min(data),
                        np.std(data),
                        skew(data),
                        kurtosis(data)])
        return out

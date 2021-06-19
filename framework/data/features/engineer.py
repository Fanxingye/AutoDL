import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
from PIL import Image
from types import SimpleNamespace
from utils.constant import Constant


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


class EngineerFeature:
    def __init__(self, task_config, csv_path=Constant.ENGINEER_FEATURES_CSV, save_to_file = False):
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
        self.data_name = task_config["data_name"]
        self.data_path = task_config["data_path"]
        self.csv_path = csv_path

        self.df = self._load_csv()
        self.entry = self._generate_feature(save_to_file)

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

import os
import math
import random
import pandas
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial import distance
from utils.constant import Constant

class DNNFeature:
    def __init__(self, task_config,
                 csv_path=Constant.BIT_FEATURES_CSV,
                 model_path="",
                 save_to_file=False):
        ''' Extract feature vector of input dataset, and compare with known datasets to determine similarity.
        Args:
            task_config: configs containing job info
            csv_path: path to the dnn feature csv file
            save_to_file: whether save current data to file, default is False

        Params:
            data_name[str]: name of the dataset
            data_path[str]: path to the dataset
            csv_path[str]: path to the csv file that contains info about previous datasets

            DIM[int]: output shape of the model
            model_path[str]: path to the model to be loaded, use the default url if empty
            BITM[keras layer]: model used to calculate features
            df[pd.DataFrame]: data loaded from csv_path
            entry[np.ndarray]: cnn meta features of current dataset
        '''
        self.data_name = task_config.get('data_name')
        self.data_path = task_config.get('data_path')
        self.csv_path = csv_path

        self.DIM = 2048
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            print("BiT-m model file does not exist, try to download.")
            # "https://tfhub.dev/google/bit/m-r50x1/1" 
            # "tfhub.dev" is blocked in China, however, "storage.googleapis.com" still works
            self.model_path = "https://storage.googleapis.com/tfhub-modules/google/bit/m-r50x1/1.tar.gz"
            
        self.BITM = hub.KerasLayer(self.model_path, trainable=False, dtype=tf.float32)
        self.BITM.build([None, None, None, 3])
        print('BiT Medium Res50 built.')

        self.df = self._load_csv()
        self.entry = self._generate_feature(save_to_file)

    def calculate_similarity_topk(self, top_k:int) -> np.ndarray:
        ''' calculate similarity between current dataset and all entries in the csv form

        Args:
            top_k: return top k most similar dataset names

        Returns:
            names of top k datasets. Ex: ["cifar10","cifar100"]

        Raises:
            ValueError: top_k out of bound
        '''

        # validate input top_k
        num_entries = len(self.df.index)
        if top_k > num_entries:
            raise ValueError(f'Expect {top_k} most similar datasets, but total count of dataset is {num_entries}')

        # calculate distance to all the known datasets
        dists = np.zeros(num_entries, dtype=np.float32)
        for i in range(num_entries):
            dists[i] = distance.cosine(self.entry, self.df.iloc[i])

        # get top_k smallest values
        top_k_index = dists.argsort()[: :1][:top_k]
        names = np.array(self.df.index)

        return names[top_k_index]

    def _generate_feature(self, save_to_file) -> np.ndarray:
        ''' generate feature vector of the dataset

        Change self.entry from none to np.ndarray

        Args:
            save_to_file: whether save file

        Returns:
            entry: 2048 features of current dataset
        '''
        if(self.data_name in self.df.index):
            print(f'{self.data_name} already in csv file so stored features will be loaded. '
                  f'Please use another name if you entered a new dataset.')
            return np.array(self.df.loc[self.data_name])

        # extract features
        entry = self._get_deep_features(self.data_path)

        # check save to file
        if save_to_file:
            df = pd.DataFrame(entry, index=[self.data_name])
            df.to_csv(self.csv_path, mode='a', header=False)
        return entry

    def _load_csv(self) -> pandas.DataFrame:
        ''' Load a csv file of dnn features

        csv file should have dataset name as index label.
        csv file should have shape of n * 2048, where n is number of datasets

        Args:
            param csv_path: path to csv file

        Returns:
            df: data loaded from csv file
        '''
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f'Cannot find csv file {self.csv_path}')
        df = pd.read_csv(self.csv_path, header=None, index_col=0)

        def _remove_zeros(df:pandas.DataFrame) :
            for i in df.columns :
                for j in df.index :
                    if df[i][j] < 1e-7 :
                        df[i][j] = 1e-7
            return df

        df = _remove_zeros(df)
        return df

    def _sample_num_strategy(self, mean: int, total: int) -> int :
        ''' Strategy when sampling images from datset.

        Logic:
            1. expect samples to get 10% of images of each class,

            2. expect 5%mean < samples < 10%mean,
            if less, use lower bound, if more, use upper bound.

            3. expect 10 < samples < 1000.
            Same as 2.

        Args:
            mean: mean of total images
            total: current class image count

        Returns:
            expected: numbers of samples from this class
        '''
        expected = math.ceil(total * 0.1)

        # 0.05mean <= expected  <= 0.1mean
        expected = max(expected, int(0.05 * mean))
        expected = min(expected, int(0.10 * mean))

        # 10 <= expected <= 1000
        expected = min(max(expected, 10), 1000)

        # expected <= total
        expected = min(expected, total)
        # print(f'total vs expected: {total}, {expected}')
        return expected

    def _get_feature_vector(self, im: str) -> np.ndarray:
        ''' Get feature vector of one image

        Args:
            im: path to image

        Returns:
            b : concatenated feature vectors of four models
        '''
        im = np.array(Image.open(im).resize([224, 224]).convert('RGB')).astype(np.float32)
        im = np.expand_dims(im, 0)
        # get feature vector
        b = np.array(self.BITM(im))
        # convert 0s to 1e-7 for later use
        for i in range(self.DIM) :
            if (b[0][i] < 1e-7) :
                b[0][i] = 1e-7
        return b

    def _get_deep_features(self, ddir: str) -> np.ndarray :
        ''' Get one vector of feature to one dataset

        Args:
            ddir: path to the dataset

        Returns:
            entry: feature vector of one dataset
        '''
        imPerClass = [len(os.listdir(os.path.join(ddir, i))) for i in os.listdir(ddir)]
        mean = int(np.mean(imPerClass))
        print(f'Image Per class Mean : {mean}')

        entry = np.zeros([1, self.DIM])
        total_sample = 0

        for j, c in enumerate(os.listdir(ddir)) :

            im_path = os.path.join(ddir, c)  # path to current class folder
            im_files = os.listdir(im_path)  # image names in the class folder
            total = len(im_files)

            sample_num = self._sample_num_strategy(mean, total)
            total_sample += sample_num
            index = random.sample(range(total), sample_num)
            print(f"Processing {j}th folder {c}. Sampled {sample_num} from total {total} images.")
            for i in index :
                im = os.path.join(im_path, im_files[i])
                entry += self._get_feature_vector(im)

        entry /= total_sample
        return entry

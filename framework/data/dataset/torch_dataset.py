"""Dataset implementation for specific task(s)"""
# pylint: disable=consider-using-generator
from gluoncv.auto.data.dataset import ImageClassificationDataset
from PIL import Image
try:
    import torch
    TorchDataset = torch.utils.data.Dataset
except ImportError:
    TorchDataset = object
    torch = None


class TorchImageClassificationDataset(ImageClassificationDataset):
    _metadata = ['classes', 'to_pytorch', 'show_images', 'random_split', 'IMG_COL', 'LABEL_COL']

    def to_pytorch(self):
        """Return a pytorch based iterator that returns ndarray and labels"""
        df = self.rename(columns={self.IMG_COL: "image", self.LABEL_COL: "label"}, errors='ignore')
        return _TorchImageClassificationDataset(df)


class _TorchImageClassificationDataset(TorchDataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ImageClassificationDataset
        DataFrame as ImageClassificationDataset.

    """
    def __init__(self, dataset):
        if torch is None:
            raise RuntimeError('Unable to import pytorch which is required.')
        assert isinstance(dataset, ImageClassificationDataset)
        assert 'image' in dataset.columns
        self._has_label = 'label' in dataset.columns
        self._dataset = dataset
        self.classes = self._dataset.classes

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        img = Image.open(im_path)
        label = None
        if self._has_label:
            label = self._dataset['label'][idx]
        return img, label


if __name__ == '__main__':
    import autogluon.core as ag
    import pandas as pd
    csv_file = ag.utils.download('https://autogluon.s3-us-west-2.amazonaws.com/datasets/petfinder_example.csv')
    df = pd.read_csv(csv_file)
    df.head()
    df = TorchImageClassificationDataset.from_csv(csv_file)
    df.head()
    print(df)
    image_dir = "/media/robin/DATA/datatsets/image_data/hymenoptera/images/split"
    train_data, _, _, =  TorchImageClassificationDataset.from_folders(image_dir)
    print(train_data)
import os
from gluoncv.auto.data.dataset import ImageClassificationDataset

class AutoGluonLoader(ImageClassificationDataset):
    @classmethod
    def generate_from_folder(self,root, exts=('.jpg', '.jpeg', '.png')):
        items = {'image': []}
        assert isinstance(root, str)
        root = os.path.abspath(os.path.expanduser(root))
        for filename in sorted(os.listdir(root)):
            filename = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in exts:
                continue
            items['image'].append(filename)
        return ImageClassificationDataset(items)
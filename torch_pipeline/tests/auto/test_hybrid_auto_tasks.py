from gluoncv.auto.tasks import ImageClassification
import autogluon.core as ag
from nose.tools import nottest

IMAGE_CLASS_DATASET, _, IMAGE_CLASS_TEST = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

def test_hybrid_image_classification():
    from gluoncv.auto.tasks import ImageClassification
    model = ag.Categorical('resnet18_v1b', 'resnet18')
    task = ImageClassification({'model': model, 'num_trials': 4, 'epochs': 1, 'batch_size': 8})
    classifier = task.fit(IMAGE_CLASS_DATASET)
    assert task.fit_summary().get('valid_acc', 0) > 0
    test_result = classifier.predict(IMAGE_CLASS_TEST)

if __name__ == '__main__':
    import nose
    nose.runmodule()

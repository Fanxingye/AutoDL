import math
import autogluon.core as ag
from autogluon.vision import ImagePredictor as Task
from mxnet import optimizer as optim
from autogluon.vision import ImagePredictor


@ag.obj(
    width_coefficient=ag.space.Categorical(1.1, 1.2),
    depth_coefficient=ag.space.Categorical(1.1, 1.2),
)
@ag.obj()
class Adam(optim.Adam):
    pass


optimizer = Adam(learning_rate=ag.Real(1e-2, 1e-1, log=True), wd=ag.Real(1e-5, 1e-3, log=True))

class EfficientNetB1(ag.model_zoo.EfficientNet):
    def __init__(self, width_coefficient, depth_coefficient):
        input_factor = 2.0 / width_coefficient / depth_coefficient
        input_size = math.ceil((224 * input_factor) / 32) * 32
        super().__init__(width_coefficient=width_coefficient,
                         depth_coefficient=depth_coefficient,
                         input_size=input_size)


if __name__ == '__main__':
    net = EfficientNetB1()
    predictor = ImagePredictor()
    classifier = predictor.fit(train_data=train_data, 
                                valid_data=valid_data,
                                hyperparameters={'net': net, 'optimizer': optimizer, 'epochs': 1}, 
                                search_strategy='grid',
                                ngpus_per_trial=1)



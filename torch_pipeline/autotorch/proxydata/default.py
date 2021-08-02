"""Default configs for image classification"""
# pylint: disable=bad-whitespace,missing-class-docstring
from typing import Union, Tuple
from autocfg import dataclass, field


@dataclass
class ImageClassification:
    model_name : str = 'resnet18'
    use_pretrained : bool = True
    use_gn : bool = False
    batch_norm : bool = False
    use_se : bool = False
    last_gamma : bool = False

@dataclass
class TrainCfg:
    pretrained_base : bool = True  # whether load the imagenet pre-trained base
    batch_size : int = 64
    epochs : int = 1
    base_lr : float = 0.01  # learning rate
    decay_factor : float = 0.1  # decay rate of learning rate.
    lr_decay_period : int = 0
    lr_decay_epoch : str = '2, 6'  # epochs at which learning rate decays
    lr_schedule_mode : str = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    warmup_lr : float = 0.0  # starting warmup learning rate.
    warmup_epochs : int = 0  # number of warmup epochs
    num_workers : int = 8
    weight_decay : float = 0.0001
    momentum : float = 0.9
    nesterov : bool = True
    dtype : str = 'float32'
    input_size : int = 224
    crop_ratio : float = 0.875
    data_augment : str = ''
    data_dir : str = ''
    no_wd : bool = False
    label_smoothing : bool = False
    resume_epoch : int = 0
    mixup : bool = False
    mixup_alpha : float = 0.2
    mixup_off_epoch : int = 0
    log_interval : int = 10
    mode : str = ''
    amp: bool = False
    static_loss_scale : float = 1.0
    dynamic_loss_scale : bool = False
    start_epoch : int = 0
    transfer_lr_mult : float = 0.01  # reduce the backbone lr_mult to avoid quickly destroying the features
    output_lr_mult : float = 0.1  # the learning rate multiplier for last fc layer if trained with transfer learning
    early_stop_patience : int = -1  # epochs with no improvement after which train is early stopped, negative: disabled
    early_stop_min_delta : float = 0.001  # ignore changes less than min_delta for metrics
    # the baseline value for metric, training won't stop if not reaching baseline
    early_stop_baseline : Union[float, int] = 0.0
    early_stop_max_value : Union[float, int] = 1.0  # early stop if reaching max value instantly

@dataclass
class ValidCfg:
    batch_size : int = 64
    num_workers : int = 8
    log_interval : int = 10


@dataclass
class TestCfg:
    batch_size : int = 1
    num_workers : int = 8


@dataclass
class RunTimeCfg:
    gpus : Union[Tuple, list] = (0, )  # gpu individual ids, not necessarily consecutive
    launcher : str = 'pytorch'
    cudnn_benchmark : bool = True

@dataclass
class ImageClassificationCfg:
    img_cls: ImageClassification = field(default_factory=ImageClassification)
    train: TrainCfg = field(default_factory=TrainCfg)
    valid: ValidCfg = field(default_factory=ValidCfg)
    test: TestCfg = field(default_factory=TestCfg)
    gpus:  Union[int, list] = (0)
    launcher: str = 'pytorch'
    cudnn_benchmark: bool = True
    distributed: bool = False

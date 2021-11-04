import kerastuner as kt
import logging
import logging.config
import os


'''
To download model wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.{npz|h5}

- model : choose from {BiT-S-R50x1,BiT-S-R50x3,BiT-S-R101x1,BiT-S-R101x3,BiT-S-R152x4,
    BiT-M-R50x1,BiT-M-R50x3,BiT-M-R101x1,BiT-M-R101x3,BiT-M-R152x4}, need to be downloaded at args.bit_pretrained_dir.
- lr : ([choice, float], [Default Value], [float range, or all choices])
    See function get_kt_lr for detailed decoding logic (Dict -> KerasTuner Object) 
    Examples: ('float', 3e-3,  [6e-4,  9e-3]),
                ('choice', 3e-3,  [5e-4, 5e-3, 5e-2, 5e-1]),
- batch_size : batch size tuple, always use choice, first as default
- epochs : maximum epoch to run, always use choice, first as default


- num_trials: trials to be run
- search_strategy : choose from ('random', 'hyper', 'bayesian')
- time_limit : max time to run, seconds as unit
- start_time : should be preserved as none
'''
#, 'BiT-M-R101x1', 'BiT-M-R101x3', 'BiT-M-R152x4' , 'BiT-M-R101x1'
SEARCH_SPACE = {
    'hyperparameters' : {
        'model' : ['BiT-S-R50x1', 'BiT-M-R50x1', 'BiT-M-R101x1'],
        'lr' :  ('float', 3e-3,  [6e-4,  9e-3]),
        'batch_size' : [32, 64],
        'epochs' : [50],
    },
    'tune_kwargs' : {
        'num_trials' : 30,
        'search_strategy' : 'bayesian',
        'time_limit' : 60 * 60* 2 ,
        'start_time' : None,
    }
}


# --------------------------------------------------------------
# ----------------- global var for path -------------------------
Name = None
DATA_PATH = None
PRETRAINED_PATH = None
OUTPUT_PATH = None # OUTPUT_PATH = args.output_path + args.name
CKPT_PATH = None   # ckpt dir = OUTPUT_PATH + 'best_so_far'
TRIALS_PATH = None # trials dir =  OUTPUT_PATH + 'trials'
BESTMODELFILE = 'best_config.p' # should be in dir CKPT_PATH

NUMCLASSES = None
LOGGER = None # log dir = OUTPUT_PATH, set up before training
TRAIN_SPLIT = 0.8

# --------------------------------------------------------------
# --------------------------------------------------------------


def get_kt_strategy(strategy = None):
    if strategy == 'random':
        return kt.oracles.RandomSearch
    elif strategy == 'hyper':
        return kt.oracles.Hyperband
    elif strategy == 'bayesian' :
        return kt.oracles.BayesianOptimization
    else :
        raise ValueError('Expected search_strategy to be '
                         'an instance of (random, hyper, bayesian), got: %s' % (strategy,))

def get_kt_lr(hp, lr):
    if(lr[0] == 'float'):
        return hp.Float('lr', lr[2][0], lr[2][1], default=lr[1])
    elif(lr[0] == 'choice'):
        return hp.Choice('lr', lr[2], default=lr[1])
    else :
        raise ValueError('Expected lr to be '
                         'choose from (float, choise), got: %s' % (lr[0],))



def setup_logger(args, logdir):
    """Creates and returns a fancy logger."""
      # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
      # Why is setting up proper logging so !@?#! ugly?
    os.makedirs(logdir, exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
         },
        "handlers": {
            "stderr": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "logfile": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": os.path.join(logdir, "train.log"),
                "mode": "a",
            }
        },
        "loggers": {
            "": {
                "handlers": ["stderr", "logfile"],
                "level": "DEBUG",
                "propagate": True
            },
        }
    })
    logger = logging.getLogger(__name__)
    logger.flush = lambda: [h.flush() for h in logger.handlers]
    logger.critical(args)
    return logger

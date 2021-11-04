## Big Transfer (BiT) with Keras Tuner
*by Yiran Wu*

This project is based on [ Big Transfer (BiT) ](https://github.com/google-research/big_transfer) and [Keras Tuner](https://github.com/keras-team/keras-tuner).
Refer to above repository for further Information.

## Introduction

This project is based on BiT, with keras tuner integrated.
Thus, HyperParameter Tuninig is performed through the different BiT models, so the best model with the best parameter can be found.


## Installation & Dependencies

Make sure you have `python>=3.6`.

Then run the following command.
```
pip install -r requirements.txt
```
Note that this project is based on tensorflow. If you got error related to cloudpickle, please install `cloudpickle==1.4.1`.

## Training Setup
First, download the BiT model. Again, please refer to  [ Big Transfer (BiT) ](https://github.com/google-research/big_transfer) for model selection.   
For this project, Only `BiT-M-R50x1`, `BiT-S-R50x1`, `BiT-M-R101x1` was used when implemented. `BiT-S-R101x1` could not be loaded, the specific reason is unknown,
the guess is that the weight file is somehow ruined.

Download with the following command.
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.h5
wget https://storage.googleapis.com/bit_models/BiT-S-R50x1.h5
wget https://storage.googleapis.com/bit_models/BiT-M-R101x1.h5
```

---
Second, open config.py to modify the `SEARCH_SPACE` value. Current supported hyperparameters are *model, lr, batch_size, epochs*. Also set max time limit/ max trials.   

---
Third, format your dataset as following:
```
└─dataset_name   
  ├─train   
     ├─class1   
     ├─class2   
     └─...   
  ├─test     
     ├─class1   
     ├─class2  
     └─...  
  ├─val   
     ├─class1    
     ├─class2  
     └─...  
```
If there is no test or val directory while you want a quick try, please change all `'val'/'test'` string to `'train'` through line 39-48 from `input_pipeline.py`.
This means your val/test accuracy is actually train accuracy.   
Note that images would all be converted to size of 256X256, and randomly cropped to 224X224, you can change resolution in `get_resolution_from_dataset` from `bit_hyperrule.py`
Data augmentation was not used, you can modify and add basic dataAug in `input_pipeline.py`.

---

Finally, run the script. The search would finish after max time limit/ max trials are reached, or all possible combinations have been run.
```
python train.py --name [TASK_NAME] --data_path [PATH_TO_DATA] --output_path [OUT_PATH] --pretrained_path [DOWNLOADED_MODEL]
```
All outputs would be saved to `OUT_PATH` as following:
```
└─[TASK_NAME]   
  ├─best_so_far   
     ├─best_config.p     # best configuration 
     ├─checkpoint        # best checkpoint for the configuration, if reloaded, refer best_config.p to load related model
  ├─train.log            # saved log, training details would be lost, but best config and test accuracy would be recorded.
  ├─trials               # keras tuner trial records
     └─...  
```


# imagedata_augmentation usage  

## 使用方式
增强： python data_aug.py --datapath 'datapath'  

默认增强的pipline为：
```
transform = A.Compose([
            	A.Rotate(limit=40,p=1),
                A.OneOf([
                	#A.RGBShift(p=0.3),
                	A.RandomBrightnessContrast(p=0.3),
            	],p=1),
                A.Flip(p=0.3),
            	A.RandomResizedCrop(image.shape[0],image.shape[1],scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333),p=0.3),
                ])
```
其中p为概率

## 数据集目录的结构
--datapath指向数据集目录目录的上一层，比如  
```
dataset
    --split
        ----train  
            ------class1  
                --------img1  
                --------img2  
                ....  
            ------class2  
                --------img1  
                --------img2  
                ....  
            ....  
```
        
在--datapath指向dataset一级目录。
增强后数据在dataset/split/train_dataaug下，包含了原图以及增强后的图像，总数量为(原图数量+增强倍数*原图数量)
增强后默认存在dataset/split/train_dataaug对应分类的文件夹下，命名规则为：  aug(单张图增强次数)_{32位乱码}.jpg

## 参数：
--datapath  数据集路径  
--augnum  单张图像数据增强次数  
--aug  增强数据开关  默认打开
--d   以benchmark的根目录来批量处理，这时需要将datapath指向benchmark根目录。处理这下面的所有文件夹，每个文件夹是一个单独的数据集。

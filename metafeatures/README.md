color mode 参考https://pillow.readthedocs.io/en/stable/handbook/concepts.html    
im_per_class表示每个类别图片数量，   
height表示图片高度   
width表示图片宽度   
area表示图片面积    

mode为众数，    
skew->skewness为偏度量    
kurt->kurtosis为峰度量   

range, std, skew, kurt均用来表示变量离散程度     
```
columns = ['name',                      # dataset name
           'class_count',               # total class count
           'image_count',               # total image count  
           'color_mode',                # reference to https://pillow.readthedocs.io/en/stable/handbook/concepts.html

            # image per class
           'im_per_class_mean',         # 平均每类图片数
           'im_per_class_median',       # 每类图片中位数
           'im_per_class_mode',         # 每类图片众数
           'im_per_class_min',          # 最少类图片数
           'im_per_class_max',          # 最多类图片数
           'im_per_class_range',        # 图片数区间长度
           'im_per_class_std',          # 图片数平均差
           'im_per_class_skew',         # 图片数偏度量                        
           'im_per_class_kurt',         # 图片数峰度量 

            # image height
           'height_mean',
           'height_median',
           'height_mode',
           'height_min',
           'height_max',
           'height_range',
           'height_std',
           'height_skew',
           'height_kurt',

            # image width
           'width_mean',
           'width_median',
           'width_mode',
           'width_min',
           'width_max',
           'width_range',
           'width_std',
           'width_skew',
           'width_kurt',

            # image area
           'area_mean',
           'area_median',
           'area_mode',
           'area_min',
           'area_max',
           'area_range',
           'area_std',
           'area_skew',
           'area_kurt']
```

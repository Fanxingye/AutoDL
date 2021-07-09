# Proxy Dataset Creation Report
 
 ## 主要流程
 1. 基于目标全量数据集, 保存每一个样本的**entropy loss**, 然后生成一个**entropy file**
    **entropy file** 内部格式: index    loss    label
 2. 根据之前生成, 或者已有的**entropy file**, 进行抽样 (**subsampling**)
 3. 抽样函数返回index数组, 利用这个index数组从全量数据集上抽样合成proxy dataset

 ## 项目结构
 1. 所有代码存在HPO下
    a. generate_entropy_file.py - 对应主要流程一
    b. create_proxy.py - 对应主要流程二
    c. main.py 57 - 81行 - 对应主要流程三, 使用torch dataloader, 根据返回的proxy index生成proxy dataset
 2. 所有的entropy file 都存在entropy_list下面

 ## Command Line 参数传递
 1. --proxy: 代表是否要用proxy dataset. (store true)
    这个参数的主要作用是方便进行 proxy 和全量数据集的对比实验.
 2. --original: 代表是否使用论文作者提供的entropy file, 论文作者提供了四个entropy file, 但是没有提供生成 entropy file的源代码. (store true)
    这个参数的主要作用是方便进行对比实验, 用自己生成的entropy file和用原文作者的entropy file的结果进行对比.
 3. --resnet50: 使用resnet50生成entropy file, 训练, 评估, 测试.
 4. --resnet18: 使用resnet18生成entropy file, 训练, 评估, 测试.
 5. --sampling_portion: proxy dataset与全量数据集的比例. default 0.2.
 6. --train_portion: training portion
 7. --fc_layers: 传入目标模型的全连阶层参数, 这个功能还没做, 需要根据传入的模型得到他的fc 参数
 8. --dataset: 目前只支持四种, cifar10, emotion detection, leaf classification, dog breed identification. 
 输入格式: [**cifar10, emotion_detection, leaf_classification, dog_breed_identification**]
    
    后面有需要可以在main函数开头加新的数据集, 读取所有数据集的路径都是在/data/AutoML_compete 下面

 ## 注意事项
 1. 目前所有实验和测试都是根据resnet 18进行, 如果用其他网络可能会出现fc参数不匹配, 输入数据可能不匹配其他网络.
 2. 读取train datatset的时候, 不要读split下面的train set. 生成entropy file的时候, 需要读取整个train set, 再保存整个train set下所有样本的entropy loss, 目前的策略是, 拿到proxy dataset的索引数组以后再分validation set.
 3. 在生成entropy file之后, 不能对全量数据集进行shuffle操作. 因为subsampling 返回的是一组索引, shuffle会打乱全量数据集原有的索引.
 4. argparser没有接入epochs, HPO Trials, 需要手动调

 ## Command Line Examples
 Current Working Directory: /automl/metafeatures/ProxyCreation/HPO


 >python main.py --resnet18 --dataset=dog_breed_identification --proxy --sampling_portion=0.25

 >python main.py --resnet18 --dataset=cifar10 --proxy

 >python main.py --resnet18 --dataset=leaf_classification --proxy


## 实验
### 实验思路
通过比较 proxy, full dataset的HPO结果, 来判断两组数据集的相似程度

###实验结果
数据集learning rate HPO比较
1. Emotion Detection: Proxy: 0.087225, Full: 0.091445
2. Leaf Classification: Proxy: 0.005803, Full: 0.006753

## Future Work
1. 需要根据输入网络, 调整输入数据的格式, main.py, 在16-18行 transform里修改
2. 需要根据输入网络, 调整fc layer参数
3. 目前是用一个输入网络, 生成entropy file和训练, 可以用不同的网络来生成entropy file和训练
4. 可以实验用不同的网络来生成entropy file.
5. 可以接入DART

## Reference
Accelerating Neural Architecture Search via Proxy Data
https://arxiv.org/abs/2106.04784
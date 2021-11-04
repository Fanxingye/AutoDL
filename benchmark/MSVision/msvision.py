import autogluon.core as ag
import microsoftvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import copy

image_size = (224,224)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Data preprocessing
def get_data():
    train_data=dset.CIFAR10(root='data',train=True,transform=data_transforms['train'],download=False)
    test_data=dset.CIFAR10(root='data',train=False,transform=data_transforms['val'],download=False)
    val_data, test_data = torch.utils.data.random_split(test_data, [int(len(test_data)/2), int(len(test_data)/2)])
    return train_data, test_data, val_data

def get_loader(train_data, test_data, val_data, bs):
    train_loader=torch.utils.data.DataLoader(train_data,batch_size=bs,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=bs,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=bs,shuffle=True)
    return train_loader, test_loader, val_loader

def get_num_correct(out, labels):
    return out.argmax(dim=1).eq(labels).sum().item()

def model_train(args, reporter):
    train_data, test_data, val_data = get_data()
    train_loader, test_loader, val_loader = get_loader(train_data, test_data, val_data, args.bs)
    train_len=len(train_data)
    test_len=len(test_data)
    val_len=len(val_data)
    print(train_len,test_len,val_len)

    model = microsoftvision.models.resnet50(pretrained=True)
    n_classes=len(train_data.classes)
    model.fc=nn.Linear(args.fcLen, n_classes)
    print('labels:',n_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model=model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr)
    epoch_num = args.epochs

    def train():
        total_loss = 0
        train_correct = 0
        for batch in train_loader:
            images,labels=batch
            outs=model(images.to(device))
            loss=F.cross_entropy(outs,labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            train_correct+=get_num_correct(outs,labels.to(device))
        return train_correct, total_loss

    def val(epoch):
        val_correct = 0
        highest_val_acc = 0
        best_model = model
        for batch in val_loader:
            images,labels=batch
            outs=model(images.to(device))
            val_correct+=get_num_correct(outs,labels.to(device))

        if val_correct/val_len > highest_val_acc:
            highest_val_acc = val_correct/test_len
            best_model = copy.deepcopy(model)
        
        reporter(epoch=epoch+1, accuracy=val_correct/val_len)
        return val_correct, best_model

    def test():
        test_correct = 0
        for batch in test_loader:
            images,labels=batch
            outs=best_model(images.to(device))
            test_correct+=get_num_correct(outs,labels.to(device))
        print("test_correct:",test_correct/test_len)

    def get_num_correct(out, labels):
        return out.argmax(dim=1).eq(labels).sum().item()
    
    for epoch in range(epoch_num):
        train_correct, total_loss = train()
        val_correct, best_model = val(epoch)
        print('process:',epoch," loss:",total_loss," train_correct:",train_correct/train_len, " val_correct:",val_correct/val_len)
    test()
    
@ag.args(
    fcLen = ag.space.Int(248,2048),
    lr = ag.space.Real(0.01, 0.2, log=True),
    bs = ag.space.Int(8,32),
    epochs=10,
)


def ag_train_mnist(args, reporter):
    return model_train(args, reporter)

myscheduler = ag.scheduler.FIFOScheduler(
    ag_train_mnist,
    resource={'num_cpus': 4, 'num_gpus': 1},
    num_trials=2,
    time_attr='epoch',
    reward_attr='accuracy')
print(myscheduler)

myscheduler.run()
myscheduler.join_jobs()
print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                               myscheduler.get_best_reward()))

f = open('optimal msvision architecture.txt',"w")
f.write(str(myscheduler.get_best_config()))
f.close()

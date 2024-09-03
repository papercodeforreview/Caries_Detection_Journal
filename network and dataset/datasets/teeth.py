from torchvision import datasets, transforms
from torch import tensor, long
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from util import TwoCropTransform, AverageMeter

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform0 = transforms.Compose(
    [
        transforms.Resize((300, 300), Image.BILINEAR),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        transforms.ToTensor(),
        normalize
    ]
)

transform1 = transforms.Compose(
    [
        transforms.Resize((300, 300), Image.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ]
)

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


class Caries(object):
    def __init__(self,data_type):
        data_categories = ['light/', 'medium/', 'severe/']
        dataset_file_path = [''+str(data_type)+'/' + data_category for data_category in data_categories]
        self.targets = []
        self.imgs = []
        self.data_type = data_type

        for file_path in dataset_file_path:
            data_file_names = os.listdir(file_path)
            for data_file_name in data_file_names:
                img = Image.open(file_path + data_file_name).convert("RGB")
                target = data_file_name.split('_')[0]
                self.imgs.append(img)
                self.targets.append(target)

    def __getitem__(self, idx):
        target = torch.as_tensor(int(self.targets[idx]))
        if self.data_type=='train':
            img = transform0(self.imgs[idx])
        else:
            img = transform1(self.imgs[idx])
        return img, target

    def __len__(self):
        return len(self.targets)


def Teeth(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 3
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    dst_train = datasets.ImageFolder(''+'/train', transform=transform0)
    dst_train_cl = datasets.ImageFolder(''+'/train', transform=TwoCropTransform(train_transform))
    dst_test = datasets.ImageFolder(''+'/test', transform=transform1)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test,dst_train_cl
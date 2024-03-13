import torch
from models.resnet import ResNet50, ResNet18
import numpy as np
from tqdm import tqdm
import os
import torchvision.transforms as transforms


def set_model(model_name, num_classes):
    if model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "reset50":
        model = ResNet50(num_classes=num_classes)
    return model


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct / total
    return acc


def set_transform(args):
    if args.dataset == "cifar10":
        train_data_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ]
        )

        test_data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
            ]
        )
        args.lr = 0.1
        args.milestones = [50, 75]
        args.num_workers = 4

    elif args.dataset in ["imagenet-100", "imagenet-dog"]:
        train_data_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        args.lr = 0.1
        args.milestones = [30, 60]
        args.num_workers = 16
    return args, train_data_transform, test_data_transform

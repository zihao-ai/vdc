from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import os
from PIL import Image
import torch
import logging


def get_transform(dataset, no_normalize=False):
    if dataset == "cifar10":
        if no_normalize:
            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
                    ),
                ]
            )

    elif dataset == "gtsrb":
        if no_normalize:
            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)
                    ),
                ]
            )

    elif dataset == "imagenet-100":
        if no_normalize:
            data_transform = transforms.Compose([transforms.ToTensor()])
        else:
            data_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
    return data_transform


class BD_Trainset(Dataset):
    def __init__(
        self,
        cl_imgfolder,
        bd_imgfolder,
        dataset,
        num_classes,
        bd_indices,
        target_label,
        transform=None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.bd_indices = bd_indices
        self.poison_indices = []
        self.img_paths = []
        self.labels = []
        self.target_label = target_label
        self.cl_imgfolder = cl_imgfolder
        self.bd_imgfolder = bd_imgfolder

        num_img = 0
        for class_i in range(num_classes):
            subfolder = f"{cl_imgfolder}/train_dataset/{class_i}"
            img_idxs = os.listdir(subfolder)
            img_idxs = [img_idx.split(".")[0] for img_idx in img_idxs]
            for img_id in img_idxs:
                if img_id in bd_indices[class_i]:
                    img_path = f"{bd_imgfolder}/train_dataset/{class_i}/{img_id}.png"
                    self.poison_indices.append(num_img)
                    self.img_paths.append(img_path)
                    self.labels.append(target_label)
                else:
                    img_path = f"{cl_imgfolder}/train_dataset/{class_i}/{img_id}.png"
                    self.img_paths.append(img_path)
                    self.labels.append(class_i)
                num_img += 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


class BD_Testset(Dataset):
    def __init__(
        self,
        bd_imgfolder,
        dataset,
        num_classes,
        target_label,
        transform=None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.target_label = target_label
        self.bd_imgfolder = bd_imgfolder

        for class_i in range(num_classes):
            if class_i == self.target_label:
                continue
            subfolder = f"{bd_imgfolder}/test_dataset/{class_i}"
            img_files = os.listdir(subfolder)
            for img_file in img_files:
                img_path = f"{bd_imgfolder}/test_dataset/{class_i}/{img_file}"
                self.img_paths.append(img_path)
                self.labels.append(target_label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


class Cleaned_BD_Trainset(Dataset):
    def __init__(
        self,
        cl_imgfolder,
        bd_imgfolder,
        dataset,
        num_classes,
        bd_indices,
        select_cl_indices,
        target_label,
        transform=None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.poison_indices = []
        self.img_paths = []
        self.labels = []
        self.target_label = target_label
        self.cl_imgfolder = cl_imgfolder
        self.bd_imgfolder = bd_imgfolder
        self.bd_indices = bd_indices

        num_img = 0
        num_bd = 0
        num_cl = 0
        for class_i in range(num_classes):
            subfolder = f"{cl_imgfolder}/train_dataset/{class_i}"
            img_idxs = os.listdir(subfolder)
            bd_idxs=bd_indices[class_i]
            img_idxs = [img_idx.split(".")[0] for img_idx in img_idxs]
            
            for img_id in img_idxs:
                if img_id in bd_idxs:
                    if img_id in select_cl_indices:
                        img_path = f"{bd_imgfolder}/train_dataset/{class_i}/{img_id}.png"
                        self.img_paths.append(img_path)
                        self.labels.append(target_label)
                        num_bd+=1
                    else:
                        pass
                else:
                    if img_id in select_cl_indices:
                        img_path = f"{cl_imgfolder}/train_dataset/{class_i}/{img_id}.png"
                        self.img_paths.append(img_path)
                        self.labels.append(class_i)
                        num_cl+=1
                    else:
                        pass
                
        logging.info(f"num_bd: {num_bd}, num_cl: {num_cl}")
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


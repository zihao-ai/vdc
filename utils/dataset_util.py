import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from utils.imagenet_class2name import IMAGENET_CLASSES
import torch


def get_class_name(dataset, num_class=200):
    if dataset == "cifar10":
        return [
            "airplane, aircraft",
            "automobile, car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif dataset == "imagenet-100":
        datasets = ImageFolder("data/origin/imagenet-100/train")
        classes = datasets.classes
        classes_names = [
            IMAGENET_CLASSES[datasets.classes[i]] for i in range(len(datasets.classes))
        ]
        # save label class class_name
        with open("data/origin/imagenet-100/classname.txt", "w") as f:
            for i in range(len(classes)):
                f.write(str(i) + " " + classes[i] + " " + classes_names[i] + "\n")
        return classes_names
    elif dataset == "imagenet-dog":
        datasets = ImageFolder("data/origin/imagenet-dog/train")
        classes = datasets.classes
        classes_names = [
            IMAGENET_CLASSES[datasets.classes[i]] for i in range(len(datasets.classes))
        ]
        # save label class class_name
        with open("data/origin/imagenet-dog/classname.txt", "w") as f:
            for i in range(len(classes)):
                f.write(str(i) + " " + classes[i] + " " + classes_names[i] + "\n")
        return classes_names
    else:
        print("Class Name is not implemented currently and use label directly.")
        return [str(i) for i in range(num_class)]


def get_class_number(dataset):
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "imagenet-dog":
        num_classes = 10
    elif dataset == "imagenet-100":
        num_classes = 100
    return num_classes


def get_class_to_id_dict(path):
    id_dict = get_id_dictionary(path)
    all_classes = {}
    result = {}
    for i, line in enumerate(open(path + "words.txt", "r")):
        n_id, word = line.split("\t")[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])
    return result


def get_id_dictionary(path, by_wnids=False):
    if by_wnids:
        id_dict = {}
        for i, line in enumerate(open(path + "wnids.txt", "r")):
            id_dict[line.replace("\n", "")] = i
        return id_dict
    else:
        classes = sorted(
            entry.name for entry in os.scandir(path + "/train") if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(
                f"Couldn't find any class folder in {path+'/train'}."
            )
        return {cls_name: i for i, cls_name in enumerate(classes)}


class MyDataset(Dataset):
    def __init__(self, imgfolder, dataset, num_classes, transform=None, train=True):
        self.dataset = dataset
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.imgfolder = imgfolder

        for class_i in range(num_classes):
            if train:
                subfolder = f"{imgfolder}/train_dataset/{class_i}"
            else:
                subfolder = f"{imgfolder}/test_dataset/{class_i}"
            img_files = os.listdir(subfolder)
            for img_file in img_files:
                if train:
                    img_path = f"{imgfolder}/train_dataset/{class_i}/{img_file}"
                else:
                    img_path = f"{imgfolder}/test_dataset/{class_i}/{img_file}"
                self.img_paths.append(img_path)
                self.labels.append(class_i)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

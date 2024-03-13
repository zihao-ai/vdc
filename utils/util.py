import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch


def fix_random(random_seed: int = 0) -> None:
    """
    use to fix randomness in the script, but if you do not want to replicate experiments, then remove this can speed up your code
    :param random_seed:
    :return: None
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def generate_common_qas(template_path, label_name):
    df = pd.read_csv(template_path)
    df = df[df["choose"] == 1]
    df["answer"] = label_name
    return df.to_dict("records")


def generate_class_specific_qas(template_path, label):
    df = pd.read_csv(template_path)
    # df = df[df["label"] == label and df
    # 找到df['label']=label 并且 df['choose']=1的数据
    df = df[(df["label"] == label) & (df["choose"] == 1)]
    return df.to_dict("records")


def read_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(data, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def set_logger(file_path):
    logFormatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(file_path)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.INFO)

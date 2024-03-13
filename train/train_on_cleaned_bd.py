import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import argparse
from utils.backdoor_util import BD_Trainset, BD_Testset,Cleaned_BD_Trainset
from utils.util import fix_random,read_pkl,save_pkl,set_logger
from utils.train_util import set_model, test, set_transform
from utils.dataset_util import get_class_number, MyDataset
import time
from tqdm import tqdm
import pickle
import logging
import pandas as pd

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--attack", type=str, default="badnet")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--llm", type=str, default="InstructBLIP")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--devices", default="0", type=str)
parser.add_argument("--target_label", type=int, default=0)
parser.add_argument("--bd_num_perclass", type=int, default=50)
parser.add_argument("--num_workers", default=4, type=float)
parser.add_argument("--milestones", default=[30, 60], type=list)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
fix_random(args.seed)
num_classes = get_class_number(args.dataset)

cl_imgfolder = f"data/{args.dataset}/{args.dataset}_clean"
bd_imgfolder = f"data/{args.dataset}/{args.dataset}_{args.attack}"

select_cl_indices=read_pkl(f"results/select_cl_index/{args.attack}_{args.target_label}_{args.llm}_{args.bd_num_perclass}.pkl")
bd_indices=read_pkl(f"data/data_index/{args.dataset}_bd_idx_{args.bd_num_perclass}.pkl")

save_folder=f"results/retrain_on_cleaned_dataset_{args.attack}_0_InstructBLIP_{args.bd_num_perclass}"
save_ckpt = f"{save_folder}/checkpoint.pt"
save_rescsv = f"{save_folder}/results.csv"

if not os.path.exists(os.path.dirname(save_ckpt)):
    os.makedirs(os.path.dirname(save_ckpt))

set_logger(f"{save_folder}/log.txt")



args, train_data_transform, test_data_transform = set_transform(args)

logging.info(args)


poisoned_train_set = Cleaned_BD_Trainset(
    cl_imgfolder=cl_imgfolder,
    bd_imgfolder=bd_imgfolder,
    dataset=args.dataset,
    num_classes=num_classes,
    bd_indices=bd_indices,
    select_cl_indices=select_cl_indices,
    target_label=args.target_label,
    transform=train_data_transform,
)
poisoned_testset = BD_Testset(
    bd_imgfolder=bd_imgfolder,
    dataset=args.dataset,
    num_classes=num_classes,
    target_label=args.target_label,
    transform=test_data_transform,
)
clean_testset = MyDataset(
    imgfolder=cl_imgfolder,
    dataset=args.dataset,
    num_classes=num_classes,
    transform=test_data_transform,
    train=False,
)
poisoned_trainloader = torch.utils.data.DataLoader(
    poisoned_train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
poisoned_testloader = torch.utils.data.DataLoader(
    poisoned_testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)
clean_testloader = torch.utils.data.DataLoader(
    clean_testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

model = set_model(args.model, num_classes)
model = nn.DataParallel(model)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(
    model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)


# Train
for epoch in range(args.epochs):
    start_time = time.time()
    model.train()
    preds = []
    labels = []
    for data, target in poisoned_trainloader:
        data = data.cuda()
        target = target.cuda()

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(
        "<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s".format(
            epoch, loss.item(), optimizer.param_groups[0]["lr"], elapsed_time
        )
    )
    scheduler.step()
    if epoch % 10 == 0:
        test_asr = test(model, poisoned_testloader)
        test_acc = test(model, clean_testloader)
        logging.info(f"Test ASR: {test_asr:.4f} Test ACC: {test_acc:.4f}")

test_asr = test(model, poisoned_testloader)
test_acc = test(model, clean_testloader)
logging.info(f"Test ASR: {test_asr:.4f} Test ACC: {test_acc:.4f}")
torch.save(model.module.state_dict(), save_ckpt)

# save test_asr,test_acc to csv
df=pd.DataFrame({"test_asr":[test_asr],"test_acc":[test_acc]})
df.to_csv(save_rescsv,index=False)


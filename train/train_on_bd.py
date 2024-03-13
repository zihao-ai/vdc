import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm

from utils.backdoor_util import BD_Testset, BD_Trainset
from utils.dataset_util import MyDataset, get_class_number
from utils.train_util import set_model, set_transform, test
from utils.util import fix_random

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--attack", type=str, default="badnet")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model", type=str, default="resnet18")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--devices", default="0", type=str)
parser.add_argument("--bd_num_perclass", type=int, default=50)
parser.add_argument("--target_label", type=int, default=0)
parser.add_argument("--num_workers", default=4, type=float)
parser.add_argument("--milestones", default=[30, 60], type=list)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
fix_random(args.seed)

if not os.path.exists("ckpts/models/backdoor"):
    os.makedirs("ckpts/models/backdoor")

cl_imgfolder = f"data/cifar10/{args.dataset}_clean"
bd_imgfolder = f"data/{args.dataset}/{args.dataset}_{args.attack}"
with open(
    f"data/data_index/{args.dataset}_bd_idx_{args.bd_num_perclass}.pkl", "rb"
) as f:
    bd_indices = pickle.load(f)

save_ckpt = f"results/train_bd_{args.model}_{args.dataset}_{args.attack}_{args.bd_num_perclass}.pth"
save_restxt = f"results/train_bd_{args.model}_{args.dataset}_{args.attack}_{args.bd_num_perclass}.txt"

num_classes = get_class_number(args.dataset)


args, train_data_transform, test_data_transform = set_transform(args)


poisoned_train_set = BD_Trainset(
    cl_imgfolder=cl_imgfolder,
    bd_imgfolder=bd_imgfolder,
    dataset=args.dataset,
    num_classes=num_classes,
    bd_indices=bd_indices,
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
    print(
        "<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s".format(
            epoch, loss.item(), optimizer.param_groups[0]["lr"], elapsed_time
        )
    )
    scheduler.step()
    if epoch % 10 == 0:
        test_asr = test(model, poisoned_testloader)
        test_acc = test(model, clean_testloader)
        print(f"Test ASR: {test_asr:.4f} Test ACC: {test_acc:.4f}")

test_asr = test(model, poisoned_testloader)
test_acc = test(model, clean_testloader)
print(f"Test ASR: {test_asr:.4f} Test ACC: {test_acc:.4f}")

with open(save_restxt, "w") as f:
    f.write("Test ASR: {:.8f}\n".format(test_asr))
    f.write("Test ACC: {:.8f}\n".format(test_acc))

torch.save(model.module.state_dict(), save_ckpt)

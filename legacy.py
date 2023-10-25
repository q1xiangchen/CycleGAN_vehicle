import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as M
import logging
import time
import argparse

from utils.utils import (
    set_random_seed,
    AverageMeter,
)
from utils.data_loader import (
    VehicleDataset,
    transforms_src,
    transforms_tar,
)


def train(
        model, 
        src_loader, 
        tar_loader, 
        tar_test_loader, 
        optimizer,
        criterion,
        args
):
    best_acc = 0.0

    for epoch in range(args.n_epoch):
        model.train()
        train_loss_clf = AverageMeter()
        train_loss_total = AverageMeter()

        # map the source and target data to the same length
        iter_src = iter(src_loader)
        iter_tar = iter(tar_loader)

        for _ in range(args.batch_size):
            data_src, label_src = next(iter_src)
            try:
                data_tar, _ = next(iter_tar)
            except StopIteration:
                iter_tar = iter(tar_loader)
                data_tar, _ = next(iter_tar)

            data_src, label_src = data_src.to(device), label_src.to(device)
            data_tar = data_tar.to(device)
        
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                feat_src, clf_src = model(data_src)
                feat_tar, _ = model(data_tar)

                clf_loss = criterion(clf_src, label_src)
                loss = clf_loss
                loss.backward()
                optimizer.step()


            train_loss_clf.update(clf_loss.item())
            train_loss_total.update(loss.item())

        # format in 4 decimal places
        log = f"Epoch: {epoch+1}/{args.n_epoch}, train_loss_clf: {train_loss_clf.avg:.4f}, " \
              f"train_loss_total: {train_loss_total.avg:.4f}, "
        
        # test
        test_acc, test_loss = test(model, tar_test_loader)
        logging.info(f"{log} test_acc: {test_acc:.4f}, test_loss: {test_loss:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            # best_model = copy.deepcopy(model.state_dict())
    # model.load_state_dict(best_model)  
    logging.info(f"best_acc: {best_acc:.4f}")      
    return model


def test(model, test_loader):
    model.eval()
    running_loss = AverageMeter()
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss.update(loss.item())
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    epoch_acc = 100.0 * correct / len(test_loader.dataset)
    return epoch_acc, running_loss.avg


class Baseline_ResNet50(nn.Module):
    def __init__(self, output_dim):
        super(Baseline_ResNet50, self).__init__()
        self.resnet50 = M.resnet50(weights=None)
        self.resnet50.load_state_dict(torch.load('./ckpt/resnet50-11ad3fa6.pth'))

        self.feature_extractor = nn.Sequential(*list(self.resnet50.children())[:-1])
        feat_dim = self.resnet50.fc.in_features
        self.clf_fc = nn.Linear(feat_dim, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        clf = self.clf_fc(x)
        return x, clf
      

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # *****************************************************
    # hyper parameters
    output_dim = 1362
    transfer_loss_weight = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
    args = parser.parse_args()
    # *****************************************************

    set_random_seed(args.seed)

    # data loader
    src_data = VehicleDataset("train", transforms_src)
    tar_data = VehicleDataset("val", transforms_tar)
    tar_test_data = VehicleDataset("test", transforms_tar)

    src_loader = DataLoader(src_data, args.batch_size, shuffle=True, num_workers=4)
    tar_loader = DataLoader(tar_data, args.batch_size, shuffle=True, num_workers=4)
    tar_test_loader = DataLoader(tar_test_data, args.batch_size, shuffle=True, num_workers=4) 

    # model
    model = Baseline_ResNet50(output_dim)
    model = model.to(device)

    # optimizer and criterion
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Optimizer not found!")
    
    if args.criterion == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not found!")

    # logging
    path = f"Baseline_experiment/Adam_b{args.batch_size}_lr{args.lr}"
    file_name = f"{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}"
    if not os.path.exists(f"./log/{path}"):
        os.makedirs(f"./log/{path}")
    logging.basicConfig(filename=f'./log/{path}/{file_name}.log', level=logging.INFO)
    logging.info(f"Train info: lr: {args.lr}, batch_size: {args.batch_size}, n_epoch: {args.n_epoch}, optimizer: {args.optimizer}, criterion: {args.criterion}")

    # train
    train(model, src_loader, tar_loader, tar_test_loader, optimizer, criterion, args)
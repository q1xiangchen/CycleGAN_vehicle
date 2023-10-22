import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as M
import logging
import time
from PIL import Image

from utils import (
    set_random_seed,
    AverageMeter,
    transforms_src,
    transforms_tar
)


def path_generator(type):
    paths = []
    labels = []
    # get the root path of the dataset
    type_root = os.path.join(root, type)
    finegrained_labels = os.listdir(type_root)
    # loop each label
    for label in finegrained_labels:
        label_path = os.path.join(type_root, label)
        paths.append(label_path)
        labels.append(int(label.split("_")[0])) 
    return paths, labels


def coral(source, target):
    # my implementation of the original paper, the code is different, but the result is the same
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    # calculating D'D for source and target
    cov_s = source.T @ source
    cov_t = target.T @ target

    # divide D'D by (num-1)
    cov_s = cov_s / (ns - 1)
    cov_t = cov_t / (nt - 1)

    # identity is a row vector of 1s
    identity_s = torch.ones((1, ns), device=source.device)
    identity_t = torch.ones((1, nt), device=target.device)

    # calculate the mean of D per column
    mean_s = identity_s @ source
    mean_t = identity_t @ target

    # calculate the squared mean
    square_mean_s = mean_s.T @ mean_s
    square_mean_t = mean_t.T @ mean_t

    # divide squared mean by (num*(num-1))
    square_mean_s = square_mean_s / (ns * (ns - 1))
    square_mean_t = square_mean_t / (nt * (nt - 1))

    # cov is (1/(num-1))*(D'*D) - (1/(num*(num-1)))*(mean)^T*(mean)
    cov_s = cov_s - square_mean_s
    cov_t = cov_t - square_mean_t

    # cov_s - cov_t
    diff = cov_s - cov_t

    # loss = (1/4)*(1/(dim*dim))*square_norm
    square_norm = torch.sum(torch.multiply(diff, diff))
    loss = square_norm / (4 * d * d)

    return loss


def train(
        model, 
        src_loader, 
        tar_loader, 
        tar_test_loader, 
        optimizer,
        criterion
):
    best_acc = 0.0
    # best_model = copy.deepcopy(model.state_dict())

    for epoch in range(n_epoch):
        model.train()
        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()

        iter_src, iter_tar = iter(src_loader), iter(tar_loader)

        for _ in range(batch_size):
            data_src, label_src = next(iter_src)
            data_tar, _ = next(iter_tar)
            data_src, label_src = data_src.to(device), label_src.to(device)
            data_tar = data_tar.to(device)

            out_s, clf_s = model(data_src)
            out_t, _ = model(data_tar)

            clf_loss = criterion(clf_s, label_src)
            transfer_loss = coral(out_s, out_t)
            loss = clf_loss + transfer_loss_weight * transfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

        # format in 4 decimal places
        log = f"Epoch: {epoch+1}/{n_epoch}, train_loss_clf: {train_loss_clf.avg:.4f}, " \
              f"train_loss_transfer: {train_loss_transfer.avg:.4f}, " \
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


def test(model, target_test_loader):
    model.eval()
    test_loss = AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(device), target.to(device)
            _, clf = model.forward(data)
            loss = criterion(clf, target)
            test_loss.update(loss.item())
            pred = torch.max(clf, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100.0 * correct / len_target_dataset
    return acc, test_loss.avg


class VehicleDataset(Dataset):
    def __init__(self, type, transform=None):
        paths, labels = path_generator(type)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = Image.open(self.paths[idx])
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label


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

    # hyper parameters
    seed = 42
    n_epoch = 100
    batch_size = 32
    lr = 1e-3
    output_dim = 1362
    transfer_loss_weight = 0
    root = "data/VehicleX/ReID Task/"
    set_random_seed(seed)

    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--n_epoch", type=int, default=20)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # args = parser.parse_args()

    src_data = VehicleDataset("train", transforms_src)
    tar_data = VehicleDataset("gallery", transforms_tar)
    tar_test_data = VehicleDataset("query", transforms_tar)

    src_loader = DataLoader(src_data, batch_size, shuffle=True, num_workers=4)
    tar_loader = DataLoader(tar_data, batch_size, shuffle=True, num_workers=4)
    tar_test_loader = DataLoader(tar_test_data, batch_size, shuffle=True, num_workers=4)

    model = Baseline_ResNet50(output_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    path = f"Baseline_experiment/Adam_b{batch_size}_lr{lr}"
    file_name = f"{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}"
    # logging with permission of creating new folder
    if not os.path.exists(f"./log/{path}"):
        os.makedirs(f"./log/{path}")
    logging.basicConfig(filename=f'./log/{path}/{file_name}.log', level=logging.INFO)
    logging.info(f"Train info: lr: {lr}, batch_size: {batch_size}, n_epoch: {n_epoch}, optimizer: {optimizer}, criterion: {criterion}")

    train(model, src_loader, tar_loader, tar_test_loader, optimizer, criterion)
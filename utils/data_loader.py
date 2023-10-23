import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


transforms =  T.Compose(
    [
        T.Resize([288,288], interpolation=3),
        T.RandomHorizontalFlip(),
        T.RandomCrop((224,224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


transforms_test =  T.Compose(
    [
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

def path_generator(type):
    paths = []
    labels = []
    src_root = "data/VehicleX/Classification Task"
    tar_root = "data/VehicleX/ReID Task/"

    # get the root path of the dataset
    if type == "train" or type == "test":
        type_root = os.path.join(src_root, type)
    elif type == "gallery" or type == "query":
        type_root = os.path.join(tar_root, type)
    else:
        raise ValueError("type must be train, test, gallery or query")

    files = os.listdir(type_root)
    # loop each label
    for label in files:
        label_path = os.path.join(type_root, label)
        paths.append(label_path)
        labels.append(int(label.split("_")[0])-1) 
    return paths, labels


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

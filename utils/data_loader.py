import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


transforms =  T.Compose(
    [
        T.Resize([288,288], T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomCrop((224,224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


transforms_test =  T.Compose(
    [
        T.Resize([224,224], T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

def path_generator(type):
    paths = []
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
    for file_name in files:
        label_path = os.path.join(type_root, file_name)
        paths.append(label_path)
    return paths


class VehicleDataset(Dataset):
    def __init__(self, type, transform=None):
        paths = path_generator(type)
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        feature = Image.open(self.paths[idx])
        if self.transform:
            feature = self.transform(feature)
        return feature, "unlabelled"
    

if __name__ == "__main__":
    # remove the file in the directory A with the same name as the directory B
    def remove_same_name_file(A, B):
        count = 0
        files = os.listdir(A)
        for file in files:
            file_path = os.path.join(A, file)
            if os.path.exists(os.path.join(B, file)):
                os.remove(file_path)
                count += 1
        print(f"remove {count} files in directory {A}")
        
    remove_same_name_file("data/VehicleX/ReID Task/gallery", "data/VehicleX/ReID Task/query")
    remove_same_name_file("data/VehicleX/Classification Task/train", "data/VehicleX/Classification Task/test")
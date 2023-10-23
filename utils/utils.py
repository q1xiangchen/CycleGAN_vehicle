import numpy as np
import random
import torch
import copy
import os


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cuda(xs):
    # print (torch.cuda.is_available())
    if torch.cuda.is_available():
        #     print (type (xs))
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]
        

class ItemPool(object):

    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        """`in_items` is a list of item."""
        if self.max_num <= 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items
    

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        # print (path)
        # print (type (path))
        if not isinstance(path,str):
            for p in path:
                # print (p)
                if not os.path.isdir( str(p)):
                    os.makedirs(str(p))
        else:
            if not os.path.isdir(path):
                os.makedirs(path)


def reorganize():
    src_root = "data/VehicleX/Classification Task"
    tar_root = "data/VehicleX/ReID Task/"
    dirs = {}
    dirs['train'] = os.path.join(src_root, 'train')
    dirs['test'] = os.path.join(src_root, 'test')
    dirs['gallery'] = os.path.join(tar_root, 'gallery')
    dirs['query'] = os.path.join(tar_root, 'query')
    mkdir(dirs.values())

    for key in dirs.keys():
        try:
            os.remove(os.path.join(dirs[key], '0'))
        except:
            pass

        if key in ['train', 'test']:
            dataset_dir = src_root
        else:
            dataset_dir = tar_root

        os.symlink(os.path.abspath(os.path.join(dataset_dir, key)),
                    os.path.join(dirs[key], '0'))

    return dirs
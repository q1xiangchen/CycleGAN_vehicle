import os

import torch
import torchvision
from torch.utils.data import DataLoader

import utils.utils as utils
import models.cycle_gan_model as cycle_gan
from utils.data_loader import VehicleDataset, transforms_test


if __name__ == "__main__":
    # setup model
    Gen_src = cycle_gan.Generator()
    Gen_tar = cycle_gan.Generator()

    # load model
    src_test_data = VehicleDataset("test", transform=transforms_test)
    src_test_loader = DataLoader(src_test_data, batch_size=1, shuffle=True, num_workers=0)

    # load checkpoint
    load_path = 'ckpt/cyclegan/lr_0.0002_batch_size_18_n_epoch_8_2023-10-25-14-39/'
    ckpt = utils.load_checkpoint(load_path)
    Gen_tar.load_state_dict(ckpt['Gen_tar'])

    save_dir = './data/VehicleX/ReID Task/fake_images/'
    if os.path.exists(save_dir):
        utils.del_file(save_dir)
    os.makedirs(save_dir)

    Gen_tar = Gen_tar.cuda()
    Gen_tar.eval()

    for i, a_real in enumerate(src_test_loader):
        if i % 1000 == 0:
            print(i)
        name = a_real[1]
        a_real = a_real[0].cuda()

        # generate fake images
        b_fake = Gen_tar(a_real)

        # save image files to folder
        pic = (torch.cat([b_fake], dim=0).data + 1) / 2.0
        torchvision.utils.save_image(pic, os.path.join(save_dir, f'{i}.jpg'))





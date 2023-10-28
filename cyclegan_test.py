import os

import torchvision
from torch.utils.data import DataLoader

import utils.utils as utils
import models.cycle_gan_model as cycle_gan
from utils.data_loader import VehicleDataset, transforms_test
from test_fid import calculate_fd_given_paths


if __name__ == "__main__":
    # setup model
    Gen_tar = cycle_gan.Generator()

    # load model
    src_test_data = VehicleDataset("test", transform=transforms_test)
    src_test_loader = DataLoader(src_test_data, batch_size=1, num_workers=0)

    # load checkpoint
    #TODO: modify the path
    load_path = 'ckpt/cyclegan/lr_0.0002_batch_size_18_n_epoch_8_2023-10-25-14-39/'
    ckpt = utils.load_checkpoint(load_path)
    Gen_tar.load_state_dict(ckpt['Gen_tar'])
    dirpatha, a, filenamea = os.walk('data/VehicleX/Classification Task/test').__next__()

    save_dir = './data/VehicleX/ReID Task/fake_images/'
    query_dir = './data/VehicleX/ReID Task/query/'
    utils.mkdir(save_dir)
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            os.remove(save_dir + file)

    Gen_tar = Gen_tar.cuda()
    Gen_tar.eval()

    for i, a_real in enumerate(src_test_loader):
        if i == 100: 
            print(f"Finished generating test images. Saved to {save_dir}")
            break

        a_real = a_real[0].cuda()
        # generate fake images
        b_fake = Gen_tar(a_real)
        # save image files to folder
        torchvision.utils.save_image((b_fake.data[0] + 1) / 2.0, save_dir + filenamea[i], padding=0)

    print("Calculating FID...")
    fid_value = calculate_fd_given_paths([query_dir, save_dir])
    print("FID: ", fid_value)
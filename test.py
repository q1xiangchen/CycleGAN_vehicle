import os

import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.utils as utils
import models.cycle_gan_model as cycle_gan
from utils.data_loader import VehicleDataset, transforms_test


def fid_score(real, gen):
    # FID = ||mu_real - mu_gen||^2 + Tr(sigma_real + sigma_gen - 2*sqrt(sigma_real*sigma_gen))
    mu_real = torch.mean(real, dim=0)
    mu_gen = torch.mean(gen, dim=0)

    diff_mean = torch.norm(mu_real - mu_gen)**2

    sigma_real = torch.cov(real, rowvar=False)
    sigma_gen = torch.cov(gen, rowvar=False)
    sqrt_sigma = torch.sqrt(torch.mm(sigma_real, sigma_gen))

    fid = diff_mean + torch.trace(sigma_real + sigma_gen - 2*sqrt_sigma)

    return fid


if __name__ == "__main__":
    # setup model
    Gen_src = cycle_gan.Generator()
    Gen_tar = cycle_gan.Generator()

    # load model
    src_test_data = VehicleDataset("test", transform=transforms_test)
    tar_test_data = VehicleDataset("query", transform=transforms_test)
    src_test_loader = DataLoader(src_test_data, batch_size=1, shuffle=True, num_workers=0)
    tar_test_loader = DataLoader(tar_test_data, batch_size=1, shuffle=True, num_workers=0)

    # load checkpoint
    load_path = './ckpt/cyclegan'
    ckpt = utils.load_checkpoint(load_path)
    Gen_src.load_state_dict(ckpt['Gen_src'])
    Gen_tar.load_state_dict(ckpt['Gen_tar'])

    dirpatha, a, filenamea = os.walk('../datasets/summer2winter_yosemite/testA/0').__next__()
    filenamea.sort()
    dirpathb, b, filenameb = os.walk('../datasets/summer2winter_yosemite/testB/0').__next__()
    filenameb.sort()
    exit()


    save_dir = './sample_images_while_testing/'
    utils.mkdir(save_dir)

    Gen_src = Gen_src.cuda()
    Gen_tar = Gen_tar.cuda()

    Gen_src.eval()
    Gen_tar.eval()

    for i, (a_real, b_real) in enumerate(zip(src_test_loader, tar_test_loader)):
        a_real = a_real[0]
        b_real = b_real[0]
        a_real, b_real = utils.cuda([a_real, b_real])

        # generate fake images
        a_fake = Gen_src(b_real)
        b_fake = Gen_tar(a_real)

        # reconstructions images
        a_rec = Gen_src(b_fake)
        b_rec = Gen_tar(a_fake)

        # FID score
        fid_a = fid_score(a_real, a_fake)
        fid_b = fid_score(b_real, b_fake)





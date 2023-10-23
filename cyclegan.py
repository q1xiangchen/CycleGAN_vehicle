import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import logging
import time
import argparse
import models.cycle_gan_model as cycle_gan
import numpy as np
from utils.utils import (
    set_random_seed,
    ItemPool,
    cuda,
    reorganize
)
from utils.data_loader import (
    VehicleDataset,
    transforms,
    transforms_test
)

if __name__ == "__main__":
    # *****************************************************
    # hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--n_epoch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    # *****************************************************

    set_random_seed(args.seed)

    # data loader
    # src_data = VehicleDataset("train", transforms)
    # src_test_data = VehicleDataset("test", transforms_test)
    # tar_data = VehicleDataset("gallery", transforms)
    # tar_test_data = VehicleDataset("query", transforms_test)

    dataset_dirs = reorganize()
    src_data = ImageFolder(dataset_dirs["train"], transforms)
    src_test_data = ImageFolder(dataset_dirs["test"], transforms_test)
    tar_data = ImageFolder(dataset_dirs["gallery"], transforms)
    tar_test_data = ImageFolder(dataset_dirs["query"], transforms_test)

    src_loader = DataLoader(src_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    src_test_loader = DataLoader(src_test_data, batch_size=1, shuffle=True, num_workers=0)
    tar_loader = DataLoader(tar_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    tar_test_loader = DataLoader(tar_test_data, batch_size=1, shuffle=True, num_workers=0) 

    # load first batch to check the shape of the data
    # iter_src = iter(src_loader)
    # a_real, b_real = next(iter_src)
    # print(a_real.shape)
    # print(b_real.shape)


    src_fake_pool = ItemPool()
    tar_fake_pool = ItemPool()

    # model
    Dis_src = cycle_gan.Discriminator()
    Dis_tar = cycle_gan.Discriminator()
    Gen_src = cycle_gan.Generator()
    Gen_tar = cycle_gan.Generator()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()
    Dis_src, Dis_tar, Gen_src, Gen_tar = cuda([Dis_src, Dis_tar, Gen_src, Gen_tar])

    Dis_src_opt = torch.optim.Adam(Dis_src.parameters(), lr=args.lr, betas=(0.5, 0.999))
    Dis_tar_opt = torch.optim.Adam(Dis_tar.parameters(), lr=args.lr, betas=(0.5, 0.999))
    Gen_src_opt = torch.optim.Adam(Gen_src.parameters(), lr=args.lr, betas=(0.5, 0.999))
    Gen_tar_opt = torch.optim.Adam(Gen_tar.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # logging
    path = f"cyclegan_experiment/Adam_b{args.batch_size}_lr{args.lr}"
    file_name = f"{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}"
    if not os.path.exists(f"./log/{path}"):
        os.makedirs(f"./log/{path}")
    logging.basicConfig(filename=f'./log/{path}/{file_name}.log', level=logging.INFO)
    logging.info(f"Train info: lr: {args.lr}, batch_size: {args.batch_size}, n_epoch: {args.n_epoch}")

    # train
    """ run """
    loss = {}
    for epoch in range(args.n_epoch):
        for i, (a_real, b_real) in enumerate(zip(src_loader, tar_loader)):
            if len(a_real[0]) < args.batch_size:
                print("Batch size is not matched!")
                continue

            step = epoch * min(len(src_loader), len(tar_loader)) + i + 1

            # set train
            Gen_src.train()
            Gen_tar.train()

            # leaves
            a_real = Variable(a_real[0])
            b_real = Variable(b_real[0])
            a_real, b_real = cuda([a_real, b_real])

            # train G
            a_fake = Gen_src(b_real)
            b_fake = Gen_tar(a_real)

            a_rec = Gen_src(b_fake)
            b_rec = Gen_tar(a_fake)

            # gen losses
            a_f_dis = Dis_src(a_fake)
            b_f_dis = Dis_tar(b_fake)
            r_label = cuda(Variable(torch.ones(a_f_dis.size())))
            a_gen_loss = MSE(a_f_dis, r_label)
            b_gen_loss = MSE(b_f_dis, r_label)
            
            # identity loss
            b2b = Gen_tar(b_real)
            a2a = Gen_src(a_real)
            idt_loss_b = L1(b2b, b_real)
            idt_loss_a = L1(a2a, a_real)
            idt_loss = idt_loss_a + idt_loss_b
            # rec losses
            a_rec_loss = L1(a_rec, a_real)
            b_rec_loss = L1(b_rec, b_real)
            rec_loss = a_rec_loss + b_rec_loss
            # g loss
            g_loss = a_gen_loss + b_gen_loss + rec_loss * 10.0 + 5.0 * idt_loss
            loss['G/a_gen_loss'] = a_gen_loss.item()
            loss['G/b_gen_loss'] = b_gen_loss.item()
            loss['G/rec_loss'] = rec_loss.item()
            loss['G/idt_loss'] = idt_loss.item()
            loss['G/g_loss'] = g_loss.item()
            # backward
            Gen_src.zero_grad()
            Gen_tar.zero_grad()
            g_loss.backward()
            Gen_src_opt.step()
            Gen_tar_opt.step()

            # leaves
            a_fake = Variable(torch.Tensor(src_fake_pool([a_fake.cpu().data.numpy()])[0]))
            b_fake = Variable(torch.Tensor(tar_fake_pool([b_fake.cpu().data.numpy()])[0]))
            a_fake, b_fake = cuda([a_fake, b_fake])

            # train D
            a_r_dis = Dis_src(a_real)
            a_f_dis = Dis_src(a_fake)
            b_r_dis = Dis_tar(b_real)
            b_f_dis = Dis_tar(b_fake)
            r_label = cuda(Variable(torch.ones(a_f_dis.size())))
            f_label = cuda(Variable(torch.zeros(a_f_dis.size())))

            # d loss
            a_d_r_loss = MSE(a_r_dis, r_label)
            a_d_f_loss = MSE(a_f_dis, f_label)
            b_d_r_loss = MSE(b_r_dis, r_label)
            b_d_f_loss = MSE(b_f_dis, f_label)

            a_d_loss = (a_d_r_loss + a_d_f_loss)*0.5
            b_d_loss = (b_d_r_loss + b_d_f_loss)*0.5
            loss['D/a_d_f_loss'] = a_d_f_loss.item()
            loss['D/b_d_f_loss'] = b_d_f_loss.item()
            loss['D/a_d_r_loss'] = a_d_r_loss.item()
            loss['D/b_d_r_loss'] = b_d_r_loss.item()
            # backward
            Dis_src.zero_grad()
            Dis_tar.zero_grad()
            a_d_loss.backward()
            b_d_loss.backward()
            Dis_src_opt.step()
            Dis_tar_opt.step()

            if (i + 1) % 10 == 0:
                logging.info("Epoch: %3d  Loaded: %5d/%5d" % (epoch, i + 1, min(len(src_loader), len(tar_loader))))
                logging.info("g_loss: %f  a_d_loss: %f   b_d_loss: %f" % (g_loss, a_d_loss, b_d_loss ))

            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    Gen_src.eval()
                    Gen_tar.eval()
                    a_real_test = Variable(next(iter(src_test_loader))[0])
                    b_real_test = Variable(next(iter(tar_test_loader))[0])
                    a_real_test, b_real_test = cuda([a_real_test, b_real_test])
                    # train G
                    a_fake_test = Gen_src(b_real_test)
                    b_fake_test = Gen_tar(a_real_test)

                    a_rec_test = Gen_src(b_fake_test)
                    b_rec_test = Gen_tar(a_fake_test)

                    pic = (torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1) / 2.0
                    
                    save_dir = './sample_images_while_training/cyclegan'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    vutils.save_image(pic, '%s/Epoch_%d_%dof%d.jpg' % (save_dir, epoch, i + 1, min(len(src_test_loader), len(tar_test_loader))), nrow=3)
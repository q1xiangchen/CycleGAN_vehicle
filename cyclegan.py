import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.utils as vutils

import models.cycle_gan_model as cycle_gan
from utils.data_loader import VehicleDataset, transforms, transforms_test
import utils.utils as utils
from utils.logger import Logger


if __name__ == "__main__":
    # ***************** hyper parameters *****************
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--n_epoch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--schedule", type=bool, default=True)
    args = parser.parse_args()
    # ***************** hyper parameters *****************

    utils.set_random_seed(args.seed)
    use_tensorboard = 1
    start_time = f"{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}"
    if args.schedule:
        model_name = f"schedule_lr_{args.lr}_batch_size_{args.batch_size}_n_epoch_{args.n_epoch}_{start_time}_SGD"
    else:
        model_name = f"lr_{args.lr}_batch_size_{args.batch_size}_n_epoch_{args.n_epoch}_{start_time}"

    if use_tensorboard:
        log_dir = './ckpt/cyclegan'
        utils.mkdir(log_dir)
        Logger = Logger(log_dir, model_name)

    # data loader
    src_data = VehicleDataset("train", transform=transforms)
    src_test_data = VehicleDataset("test", transform=transforms_test)
    tar_data = VehicleDataset("gallery", transform=transforms)
    tar_test_data = VehicleDataset("query", transform=transforms_test)

    src_loader = DataLoader(src_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    src_test_loader = DataLoader(src_test_data, batch_size=1, shuffle=True, num_workers=0)
    tar_loader = DataLoader(tar_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    tar_test_loader = DataLoader(tar_test_data, batch_size=1, shuffle=True, num_workers=0) 

    src_fake_pool = utils.ItemPool()
    tar_fake_pool = utils.ItemPool()

    # model
    Dis_src = cycle_gan.Discriminator()
    Dis_tar = cycle_gan.Discriminator()
    Gen_src = cycle_gan.Generator()
    Gen_tar = cycle_gan.Generator()
    MSE = nn.MSELoss()
    L1 = nn.L1Loss()
    Dis_src, Dis_tar, Gen_src, Gen_tar = utils.cuda([Dis_src, Dis_tar, Gen_src, Gen_tar])

    Dis_src_opt = torch.optim.SGD(Dis_src.parameters(), lr=args.lr, momentum=0.9)
    Dis_tar_opt = torch.optim.SGD(Dis_src.parameters(), lr=args.lr, momentum=0.9)
    Gen_src_opt = torch.optim.SGD(Dis_src.parameters(), lr=args.lr, momentum=0.9)
    Gen_tar_opt = torch.optim.SGD(Dis_src.parameters(), lr=args.lr, momentum=0.9)

    # calculate the total steps
    total_steps = args.n_epoch * min(len(src_loader), len(tar_loader))
    step_size = total_steps // 4
    gamma = 0.25
    print("Total steps: %d, step size: %d, gamma: %f" % (total_steps, step_size, gamma))

    # set scheduler
    if args.schedule:
        Dis_src_scheduler = torch.optim.lr_scheduler.StepLR(Dis_src_opt, step_size=step_size, gamma=gamma)
        Dis_tar_scheduler = torch.optim.lr_scheduler.StepLR(Dis_tar_opt, step_size=step_size, gamma=gamma)
        Gen_src_scheduler = torch.optim.lr_scheduler.StepLR(Gen_src_opt, step_size=step_size, gamma=gamma)
        Gen_tar_scheduler = torch.optim.lr_scheduler.StepLR(Gen_tar_opt, step_size=step_size, gamma=gamma)
        

    """ load checkpoint """
    ckpt_dir = './ckpt/cyclegan/' + model_name
    utils.mkdir(ckpt_dir)
    try:
        ckpt = utils.load_checkpoint(ckpt_dir)
        start_epoch = ckpt['epoch'] + 1
        Dis_src.load_state_dict(ckpt['Dis_src'])
        Dis_tar.load_state_dict(ckpt['Dis_tar'])
        Gen_src.load_state_dict(ckpt['Gen_src'])
        Gen_tar.load_state_dict(ckpt['Gen_tar'])
        Dis_src_opt.load_state_dict(ckpt['Dis_src_opt'])
        Dis_tar_opt.load_state_dict(ckpt['Dis_tar_opt'])
        Gen_src_opt.load_state_dict(ckpt['Gen_src_opt'])
        Gen_tar_opt.load_state_dict(ckpt['Gen_tar_opt'])
    except:
        print(' [*] No checkpoint!')
        start_epoch = 0

    # train
    """ run """
    loss = {}
    for epoch in range(start_epoch, args.n_epoch):
        print(f"Epoch {epoch+1}/{args.n_epoch}")
        src_test_iter, tar_test_iter = iter(src_test_loader), iter(tar_test_loader)
        for i, (a_real, b_real) in enumerate(zip(src_loader, tar_loader)): 
            if len(a_real[0]) < args.batch_size or len(b_real[0]) < args.batch_size:
                print("Batch size is not matched!")
                continue

            step = epoch * min(len(src_loader), len(tar_loader)) + i + 1
            loss['lr'] = Gen_src_opt.param_groups[0]['lr']

            # set train
            Gen_src.train()
            Gen_tar.train()

            # leaves
            a_real = Variable(a_real[0])
            b_real = Variable(b_real[0])
            a_real, b_real = utils.cuda([a_real, b_real])

            # train G
            a_fake = Gen_src(b_real)
            b_fake = Gen_tar(a_real)

            a_rec = Gen_src(b_fake)
            b_rec = Gen_tar(a_fake)

            # gen losses
            a_f_dis = Dis_src(a_fake)
            b_f_dis = Dis_tar(b_fake)
            r_label = torch.ones(a_f_dis.size()).cuda()
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
            a_fake, b_fake = utils.cuda([a_fake, b_fake])

            # train D
            a_r_dis = Dis_src(a_real)
            a_f_dis = Dis_src(a_fake)
            b_r_dis = Dis_tar(b_real)
            b_f_dis = Dis_tar(b_fake)
            r_label = torch.ones(a_f_dis.size()).cuda()
            f_label = torch.zeros(a_f_dis.size()).cuda()

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

            if args.schedule:
                Dis_src_scheduler.step()
                Dis_tar_scheduler.step()
                Gen_src_scheduler.step()
                Gen_tar_scheduler.step()

            if (i + 1) % 10 == 0:
                if use_tensorboard:
                    for tag, value in loss.items():
                        Logger.scalar_summary(tag, value, step)
            
            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    Gen_src.eval()
                    Gen_tar.eval()
                    a_real_test, b_real_test = utils.cuda([next(src_test_iter)[0], next(tar_test_iter)[0]])

                    # generate fake images
                    a_fake_test = Gen_src(b_real_test)
                    b_fake_test = Gen_tar(a_real_test)

                    # reconstruction images
                    a_rec_test = Gen_src(b_fake_test)
                    b_rec_test = Gen_tar(a_fake_test)

                    pic = (torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1) / 2.0
                    
                    save_dir = f"./sample_images_while_training/{model_name}/Epoch_{epoch+1}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    vutils.save_image(pic, '%s/%dof%d.jpg' % (save_dir, i + 1, min(len(src_loader), len(tar_loader))), nrow=3)
                
        # save checkpoint
        utils.save_checkpoint({'epoch': epoch,
                'Dis_src': Dis_src.state_dict(),
                'Dis_tar': Dis_tar.state_dict(),
                'Gen_src': Gen_src.state_dict(),
                'Gen_tar': Gen_tar.state_dict(),
                'Dis_src_opt': Dis_src_opt.state_dict(),
                'Dis_tar_opt': Dis_tar_opt.state_dict(),
                'Gen_src_opt': Gen_src_opt.state_dict(),
                'Gen_tar_opt': Gen_tar_opt.state_dict()},
                '%s/Epoch_%d.ckpt' % (ckpt_dir, epoch + 1), 
                max_keep=4)
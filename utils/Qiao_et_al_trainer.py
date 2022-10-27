import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import numpy as np
import time
from tqdm import tqdm
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import pandas as pd

class F_Trainier(nn.Module):
    def __init__(self, save_model_weights_path, save_val_results_path, fold, device, lr, num_epochs): 
        super(F_Trainier, self).__init__()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        incepv3 = InceptionV3([block_idx])
        self.incepv3 = incepv3.to(device)
        '''
        torch.save({
            'epoch': epoch,
            'fd_state_dict': self.FD.state_dict(),
            'lsd_state_dict': self.LSD.state_dict(),
            'fg_state_dict': self.FG.state_dict(),
            'fr_state_dict': self.FR.state_dict(),
            'discriminator_state_dict': self.dis_model.state_dict(),
        }, os.path.join(self.model_weights_save, 'model_' + str(epoch) + '.pth'))
        '''
        if fold == 'fold1':
            FD = LSD_FD_model(3, 1)
            LSD = LSD_FD_model(3, 1)
            FG = FR_FG_model(9, 4, 3)
            FR = FR_FG_model(9, 4, 3)
            checkpoint = torch.load(os.path.join('/workspace/DB1_proposed_model_fd_fold1_weights', 'model_168.pth'))#Best loss index: 168 Best loss:  8.011459194676718e-05
            FD.load_state_dict(checkpoint['model_state_dict'])
            LSD.load_state_dict(checkpoint['model_state_dict'])
            FG.load_state_dict(checkpoint['model_state_dict'])
            FR.load_state_dict(checkpoint['model_state_dict'])
            print('fold1 checkpoint loading complete...')
        elif fold == 'fold2':
            FD = LSD_FD_model(3, 1)
            LSD = LSD_FD_model(3, 1)
            FG = FR_FG_model(9, 4, 3)
            FR = FR_FG_model(9, 4, 3)
            checkpoint = torch.load(os.path.join('/workspace/DB1_proposed_model_fd_fold2_weights', 'model_191.pth'))#Best loss index: 191 Best loss:  0.0003442429820714616
            FD.load_state_dict(checkpoint['model_state_dict'])
            LSD.load_state_dict(checkpoint['model_state_dict'])
            FG.load_state_dict(checkpoint['model_state_dict'])
            FR.load_state_dict(checkpoint['model_state_dict'])
            print('fold2 checkpoint loading complete...')
        elif fold == 'default':
            FD = LSD_FD_model(3, 1)
            LSD = LSD_FD_model(3, 1)
            FG = FR_FG_model(9, 4, 3)
            FR = FR_FG_model(9, 4, 3)
        
        discriminator = PatchGANDiscriminator(3)

        self.FD = FD.to(device)
        self.LSD = LSD.to(device)
        self.FG = FG.to(device)
        self.FR = FR.to(device)
        self.dis_model = discriminator.to(device)
        
        save_root = '/workspace'
        model_weights_save = os.path.join(save_root, save_model_weights_path)
        os.makedirs(model_weights_save, exist_ok=True)

        result_save = os.path.join(save_root, save_val_results_path)
        os.makedirs(result_save, exist_ok=True)
        
        self.model_weights_save = model_weights_save
        self.save_val_results_path = result_save

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        
        self.gen_optimizer = optim.Adam(list(self.FR.parameters()) + list(self.FG.parameters()) + list(self.LSD.parameters()) + list(self.FD.parameters()), lr=self.lr, betas=(0.5, 0.9999))
        self.dis_optimizer = optim.Adam(self.dis_model.parameters(), lr=self.lr*0.1)

        self.l1_loss = nn.L1Loss().to(self.device)
        self.l2_loss = nn.MSELoss().to(self.device)
        self.perceptual = PerceptualLoss_Qiao().to(self.device)

        self.Training_history = {'gen_loss': [], 'dis_loss': [], 'psnr': [], 'ssim': []}
        self.Validation_history = {'gen_loss': [], 'dis_loss': [], 'psnr': [], 'ssim': [], 'fid': []}

    def eval_step(self, engine, batch):
        return batch

    def train(self, train_loader, val_loader):
        for epoch in range(1, self.num_epochs+1):
            print("Epoch {}/{} Start......".format(epoch, self.num_epochs))
            epoch_dis_loss, epoch_gen_loss, epoch_psnr, epoch_ssim = self._train_epoch(epoch, train_loader)
            epoch_val_dis_loss, epoch_val_gen_loss, epoch_val_psnr, epoch_val_ssim = self._valid_epoch(epoch, val_loader)

        self.Training_history = pd.DataFrame.from_dict(self.Training_history, orient='index')
        self.Training_history.to_csv(os.path.join(self.model_weights_save, 'train_history.csv'))
        self.Validation_history = pd.DataFrame.from_dict(self.Validation_history, orient='index')
        self.Validation_history.to_csv(os.path.join(self.model_weights_save, 'valid_history.csv'))

        print('Best PSNR score index:', self.Validation_history.loc['psnr'].idxmax() + 1, 'Best PSNR score:', self.Validation_history.loc['psnr'].max())
        print('Best SSIM score index:', self.Validation_history.loc['ssim'].idxmax() + 1, 'Best SSIM score:', self.Validation_history.loc['ssim'].max())
        print('Best FID score index:', self.Validation_history.loc['fid'].idxmin() + 1, 'Best FID score:', self.Validation_history.loc['fid'].min())

    def _train_epoch(self, epoch, train_loader):
        epoch_start_time = time.time()
        default_evaluator = Engine(self.eval_step)

        metric_ssim = SSIM(1.0)
        metric_ssim.attach(default_evaluator, 'ssim')

        metric_psnr = PSNR(1.0)
        metric_psnr.attach(default_evaluator, 'psnr')
        
        self.FD.train()
        self.LSD.train()
        self.FG.train()
        self.FR.train()
        self.dis_model.train()
        
        epoch_dis_loss = 0
        epoch_dis_real_loss = 0
        epoch_dis_fake_loss = 0
        epoch_gen_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        print("======> Train Start")
        for i, data in enumerate(tqdm(train_loader), 0):
            input = data[0].to(self.device)
            clean = data[1].to(self.device)
            
            Mf_ls = self.LSD(input) # flare image light source mask
            Mf_f = self.FD(input) # flare image flare region mask
            fr_output = self.FR(torch.cat((input, Mf_f), dim=1)) 
            Ms_hat_ls = self.LSD(fr_output) # flare removal image light source mask
            fg_output = self.FG(torch.cat((fr_output, Mf_ls), dim=1))
            Mf_hat_f = self.FD(fg_output) # flare generation image flare region mask

            # Discriminator
            self.dis_optimizer.zero_grad()

            dis_real = self.dis_model(clean)
            dis_fake = self.dis_model(fr_output.detach())

            dis_real_loss = self.l2_loss(dis_real, torch.ones_like(dis_real))
            dis_fake_loss = self.l2_loss(dis_fake, torch.zeros_like(dis_fake))

            dis_loss = dis_real_loss + dis_fake_loss

            dis_loss.backward()
            self.dis_optimizer.step()

            epoch_dis_loss += dis_loss.item()
            epoch_dis_real_loss += dis_real_loss.item()
            epoch_dis_fake_loss += dis_fake_loss.item()

            # Generator
            self.gen_optimizer.zero_grad()

            gen_fake = self.dis_model(fr_output)
            gen_gan_loss = self.l2_loss(gen_fake, torch.ones_like(gen_fake))

            ls_loss = self.l1_loss(Mf_ls, Ms_hat_ls)
            f_loss = self.l1_loss(Mf_f, Mf_hat_f)

            cycle_loss = self.l1_loss(input, fg_output) + self.perceptual(input, fg_output)

            gen_loss = ls_loss + gen_gan_loss + f_loss + cycle_loss

            gen_loss.backward()
            self.gen_optimizer.step()

            epoch_gen_loss += gen_loss.item()

            metric_state = default_evaluator.run([[fr_output, clean]])

            epoch_psnr += metric_state.metrics['psnr']
            epoch_ssim += metric_state.metrics['ssim']
        
        # self.gen_lr_scheduler.step()
        # self.dis_lr_scheduler.step()
        
        print('Epoch: {}\tTime: {:.4f}\tGen Loss: {:.4f}\tDis Loss: {:.4f}\tPSNR: {:.4f}\tSSIM: {:.4f}\tGen LR: {:.8f}\tDis LR: {:.8f}'.format(
                epoch, time.time() - epoch_start_time, epoch_gen_loss / len(train_loader), epoch_dis_loss / len(train_loader), 
                epoch_psnr / len(train_loader), epoch_ssim / len(train_loader),
                self.lr, self.lr))#self.gen_lr_scheduler.get_last_lr()[0], self.dis_lr_scheduler.get_last_lr()[0]))
        print('Dis real Loss: {:.4f}\tDis fake Loss: {:.4f}'.format(
            epoch_dis_real_loss / len(train_loader), epoch_dis_fake_loss / len(train_loader)
        ))
        self.Training_history['gen_loss'].append(epoch_gen_loss / len(train_loader))
        self.Training_history['dis_loss'].append(epoch_dis_loss / len(train_loader))
        self.Training_history['psnr'].append(epoch_psnr / len(train_loader))
        self.Training_history['ssim'].append(epoch_ssim / len(train_loader))

        torch.save({
            'epoch': epoch,
            'fd_state_dict': self.FD.state_dict(),
            'lsd_state_dict': self.LSD.state_dict(),
            'fg_state_dict': self.FG.state_dict(),
            'fr_state_dict': self.FR.state_dict(),
            'discriminator_state_dict': self.dis_model.state_dict(),
        }, os.path.join(self.model_weights_save, 'model_' + str(epoch) + '.pth'))

        return epoch_dis_loss, epoch_gen_loss, epoch_psnr, epoch_ssim

    def _valid_epoch(self, epoch, val_loader):
        self.FD.eval()
        self.LSD.eval()
        self.FG.eval()
        self.FR.eval()
        self.dis_model.eval()

        with torch.no_grad():
            val_epoch_start_time = time.time()
            val_default_evaluator = Engine(self.eval_step)

            val_metric_ssim = SSIM(1.0)
            val_metric_ssim.attach(val_default_evaluator, 'ssim')

            val_metric_psnr = PSNR(1.0) 
            val_metric_psnr.attach(val_default_evaluator, 'psnr')

            epoch_val_dis_loss = 0
            epoch_val_gen_loss = 0
            epoch_val_psnr = 0
            epoch_val_ssim = 0
            epoch_val_fid = 0

            print('======> Validation Start')
            for i, data in enumerate(tqdm(val_loader)):
                val_input = data[0].to(self.device)
                val_clean = data[1].to(self.device)

                Mf_ls = self.LSD(val_input) # flare image light source mask
                Mf_f = self.FD(val_input) # flare image flare region mask
                fr_output = self.FR(torch.cat((val_input, Mf_f), dim=1)) 
                Ms_hat_ls = self.LSD(fr_output) # flare removal image light source mask
                fg_output = self.FG(torch.cat((fr_output, Mf_ls), dim=1))
                Mf_hat_f = self.FD(fg_output) # flare generation image flare region mask
 
                # Discriminator
                dis_real = self.dis_model(val_clean)
                dis_fake = self.dis_model(fr_output.detach())

                dis_real_loss = self.l2_loss(dis_real, torch.ones_like(dis_real))
                dis_fake_loss = self.l2_loss(dis_fake, torch.zeros_like(dis_fake))

                dis_loss = dis_real_loss + dis_fake_loss

                epoch_val_dis_loss += dis_loss.item()

                # Generator
                gen_fake = self.dis_model(fr_output)
                gen_gan_loss = self.l2_loss(gen_fake, torch.ones_like(gen_fake))

                ls_loss = self.l1_loss(Mf_ls, Ms_hat_ls)
                f_loss = self.l1_loss(Mf_f, Mf_hat_f)

                cycle_loss = self.l1_loss(val_input, fg_output) + self.perceptual(val_input, fg_output)

                epoch_val_fid += val_fid
            
            torchvision.utils.save_image(torch.cat([val_input, fr_output, val_clean]), os.path.join(self.save_val_results_path, 'results_' + str(epoch) + '.png'))
            print('Epoch: {}\tTime: {:.4f}\tGen Loss: {:.4f}\tDis Loss: {:.4f}\tPSNR: {:.4f}\tSSIM: {:.4f}\tFID: {:.8f}'.format(
                    epoch, time.time() - val_epoch_start_time, epoch_val_gen_loss / len(val_loader), epoch_val_dis_loss / len(val_loader), 
                    epoch_val_psnr / len(val_loader), epoch_val_ssim / len(val_loader), epoch_val_fid / len(val_loader)
                    ))#self.gen_lr_scheduler.get_last_lr()[0], self.dis_lr_scheduler.get_last_lr()[0]))

            self.Validation_history['gen_loss'].append(epoch_val_gen_loss / len(val_loader))
            self.Validation_history['dis_loss'].append(epoch_val_dis_loss / len(val_loader))
            self.Validation_history['psnr'].append(epoch_val_psnr / len(val_loader))
            self.Validation_history['ssim'].append(epoch_val_ssim / len(val_loader))
            self.Validation_history['fid'].append(epoch_val_fid / len(val_loader))

        return epoch_val_dis_loss, epoch_val_gen_loss, epoch_val_psnr, epoch_val_ssim

class FF_Trainier(nn.Module):
    def __init__(self, save_model_weights_path, save_val_results_path, device, lr, num_epochs): 
        super(FF_Trainier, self).__init__()

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        incepv3 = InceptionV3([block_idx])
        self.incepv3 = incepv3.to(device)

        FD = LSD_FD_model(3, 1)
        LSD = LSD_FD_model(3, 1)
        FG = FR_FG_model(9, 4, 3)
        FR = FR_FG_model(9, 4, 3)
        discriminator = PatchGANDiscriminator(3)

        self.FD = FD.to(device)
        self.LSD = LSD.to(device)
        self.FG = FG.to(device)
        self.FR = FR.to(device)
        self.dis_model = discriminator.to(device)
        
        save_root = '/workspace'
        model_weights_save = os.path.join(save_root, save_model_weights_path)
        os.makedirs(model_weights_save, exist_ok=True)

        result_save = os.path.join(save_root, save_val_results_path)
        os.makedirs(result_save, exist_ok=True)
        
        self.model_weights_save = model_weights_save
        self.save_val_results_path = result_save

        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        
        self.gen_optimizer = optim.Adam(list(self.FR.parameters()) + list(self.FG.parameters()) + list(self.LSD.parameters()) + list(self.FD.parameters()), lr=self.lr, betas=(0.5, 0.9999))
        self.dis_optimizer = optim.Adam(self.dis_model.parameters(), lr=self.lr*0.1)

        self.l1_loss = nn.L1Loss().to(self.device)
        self.l2_loss = nn.MSELoss().to(self.device)
        self.perceptual = PerceptualLoss_Qiao().to(self.device)

        self.Training_history = {'gen_loss': [], 'dis_loss': [], 'psnr': [], 'ssim': []}
        self.Validation_history = {'gen_loss': [], 'dis_loss': [], 'psnr': [], 'ssim': [], 'fid': []}

    def eval_step(self, engine, batch):
        return batch

    def train(self, train_loader, val_loader):
        for epoch in range(1, self.num_epochs+1):
            print("Epoch {}/{} Start......".format(epoch, self.num_epochs))
            epoch_dis_loss, epoch_gen_loss, epoch_psnr, epoch_ssim = self._train_epoch(epoch, train_loader)
            epoch_val_dis_loss, epoch_val_gen_loss, epoch_val_psnr, epoch_val_ssim = self._valid_epoch(epoch, val_loader)

        self.Training_history = pd.DataFrame.from_dict(self.Training_history, orient='index')
        self.Training_history.to_csv(os.path.join(self.model_weights_save, 'train_history.csv'))
        self.Validation_history = pd.DataFrame.from_dict(self.Validation_history, orient='index')
        self.Validation_history.to_csv(os.path.join(self.model_weights_save, 'valid_history.csv'))

        print('Best PSNR score index:', self.Validation_history.loc['psnr'].idxmax() + 1, 'Best PSNR score:', self.Validation_history.loc['psnr'].max())
        print('Best SSIM score index:', self.Validation_history.loc['ssim'].idxmax() + 1, 'Best SSIM score:', self.Validation_history.loc['ssim'].max())
        print('Best FID score index:', self.Validation_history.loc['fid'].idxmin() + 1, 'Best FID score:', self.Validation_history.loc['fid'].min())

    def _train_epoch(self, epoch, train_loader):
        epoch_start_time = time.time()
        default_evaluator = Engine(self.eval_step)

        metric_ssim = SSIM(1.0)
        metric_ssim.attach(default_evaluator, 'ssim')

        metric_psnr = PSNR(1.0)
        metric_psnr.attach(default_evaluator, 'psnr')
        
        self.FD.train()
        self.LSD.train()
        self.FG.train()
        self.FR.train()
        self.dis_model.train()
        
        epoch_dis_loss = 0
        epoch_dis_real_loss = 0
        epoch_dis_fake_loss = 0
        epoch_gen_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        print("======> Train Start")
        for i, data in enumerate(tqdm(train_loader), 0):
            input = data[0].to(self.device)
            clean = data[1].to(self.device)
            
            Mff_ls = self.LSD(clean) # flare free image light source mask
            fg_output = self.FG(torch.cat((clean, Mff_ls), dim=1))
            Mf_hat_f = self.FD(fg_output) # flare generation image flare region mask
            fr_output = self.FR(torch.cat((fg_output, Mf_hat_f), dim=1))
            Mf_hat_ls = self.LSD(fr_output) # flare removal image light source mask
  

            # Discriminator
            self.dis_optimizer.zero_grad()

            dis_real = self.dis_model(input)
            dis_fake = self.dis_model(fg_output.detach())

            dis_real_loss = self.l2_loss(dis_real, torch.ones_like(dis_real))
            dis_fake_loss = self.l2_loss(dis_fake, torch.zeros_like(dis_fake))

            dis_loss = dis_real_loss + dis_fake_loss

            dis_loss.backward()
            self.dis_optimizer.step()

            epoch_dis_loss += dis_loss.item()
            epoch_dis_real_loss += dis_real_loss.item()
            epoch_dis_fake_loss += dis_fake_loss.item()

            # Generator
            self.gen_optimizer.zero_grad()

            gen_fake = self.dis_model(fg_output)
            gen_gan_loss = self.l2_loss(gen_fake, torch.ones_like(gen_fake))

            ls_loss = self.l1_loss(Mff_ls, Mf_hat_ls)

            cycle_loss = self.l1_loss(clean, fr_output) + self.perceptual(clean, fr_output)

            gen_loss = ls_loss + gen_gan_loss + cycle_loss

            gen_loss.backward()
            self.gen_optimizer.step()

            epoch_gen_loss += gen_loss.item()

            metric_state = default_evaluator.run([[fg_output, input]])

            epoch_psnr += metric_state.metrics['psnr']
            epoch_ssim += metric_state.metrics['ssim']
        
        # self.gen_lr_scheduler.step()
        # self.dis_lr_scheduler.step()
        
        print('Epoch: {}\tTime: {:.4f}\tGen Loss: {:.4f}\tDis Loss: {:.4f}\tPSNR: {:.4f}\tSSIM: {:.4f}\tGen LR: {:.8f}\tDis LR: {:.8f}'.format(
                epoch, time.time() - epoch_start_time, epoch_gen_loss / len(train_loader), epoch_dis_loss / len(train_loader), 
                epoch_psnr / len(train_loader), epoch_ssim / len(train_loader),
                self.lr, self.lr))#self.gen_lr_scheduler.get_last_lr()[0], self.dis_lr_scheduler.get_last_lr()[0]))
        print('Dis real Loss: {:.4f}\tDis fake Loss: {:.4f}'.format(
            epoch_dis_real_loss / len(train_loader), epoch_dis_fake_loss / len(train_loader)
        ))
        self.Training_history['gen_loss'].append(epoch_gen_loss / len(train_loader))
        self.Training_history['dis_loss'].append(epoch_dis_loss / len(train_loader))
        self.Training_history['psnr'].append(epoch_psnr / len(train_loader))
        self.Training_history['ssim'].append(epoch_ssim / len(train_loader))

        torch.save({
            'epoch': epoch,
            'fd_state_dict': self.FD.state_dict(),
            'lsd_state_dict': self.LSD.state_dict(),
            'fg_state_dict': self.FG.state_dict(),
            'fr_state_dict': self.FR.state_dict(),
            'discriminator_state_dict': self.dis_model.state_dict(),
        }, os.path.join(self.model_weights_save, 'model_' + str(epoch) + '.pth'))

        return epoch_dis_loss, epoch_gen_loss, epoch_psnr, epoch_ssim

    def _valid_epoch(self, epoch, val_loader):
        self.FD.eval()
        self.LSD.eval()
        self.FG.eval()
        self.FR.eval()
        self.dis_model.eval()

        with torch.no_grad():
            val_epoch_start_time = time.time()
            val_default_evaluator = Engine(self.eval_step)

            val_metric_ssim = SSIM(1.0)
            val_metric_ssim.attach(val_default_evaluator, 'ssim')

            val_metric_psnr = PSNR(1.0) 
            val_metric_psnr.attach(val_default_evaluator, 'psnr')

            epoch_val_dis_loss = 0
            epoch_val_gen_loss = 0
            epoch_val_psnr = 0
            epoch_val_ssim = 0
            epoch_val_fid = 0

            print('======> Validation Start')
            for i, data in enumerate(tqdm(val_loader)):
                val_input = data[0].to(self.device)
                val_clean = data[1].to(self.device)

                Mff_ls = self.LSD(val_clean) # flare free image light source mask
                fg_output = self.FG(torch.cat((val_clean, Mff_ls), dim=1))
                Mf_hat_f = self.FD(fg_output) # flare generation image flare region mask
                fr_output = self.FR(torch.cat((fg_output, Mf_hat_f), dim=1))
                Mf_hat_ls = self.LSD(fr_output) # flare removal image light source mask

                # Discriminator
                dis_real = self.dis_model(val_input)
                dis_fake = self.dis_model(fg_output.detach())

                dis_real_loss = self.l2_loss(dis_real, torch.ones_like(dis_real))
                dis_fake_loss = self.l2_loss(dis_fake, torch.zeros_like(dis_fake))

                dis_loss = dis_real_loss + dis_fake_loss

                epoch_val_dis_loss += dis_loss.item()

                # Generator
                gen_fake = self.dis_model(fg_output)
                gen_gan_loss = self.l2_loss(gen_fake, torch.ones_like(gen_fake))

                ls_loss = self.l1_loss(Mff_ls, Mf_hat_ls)

                cycle_loss = self.l1_loss(val_clean, fr_output) + self.perceptual(val_clean, fr_output)

                gen_loss = ls_loss + gen_gan_loss + cycle_loss

                epoch_val_gen_loss += gen_loss.item()

                metric_state = val_default_evaluator.run([[fg_output, val_input]])

                epoch_val_psnr += metric_state.metrics['psnr']
                epoch_val_ssim += metric_state.metrics['ssim']
                val_fid = calculate_fretchet(fg_output, val_input, self.incepv3)
                epoch_val_fid += val_fid
            
            torchvision.utils.save_image(torch.cat([val_clean, fg_output, val_input]), os.path.join(self.save_val_results_path, 'results_' + str(epoch) + '.png'))
            print('Epoch: {}\tTime: {:.4f}\tGen Loss: {:.4f}\tDis Loss: {:.4f}\tPSNR: {:.4f}\tSSIM: {:.4f}\tFID: {:.8f}'.format(
                    epoch, time.time() - val_epoch_start_time, epoch_val_gen_loss / len(val_loader), epoch_val_dis_loss / len(val_loader), 
                    epoch_val_psnr / len(val_loader), epoch_val_ssim / len(val_loader), epoch_val_fid / len(val_loader)
                    ))#self.gen_lr_scheduler.get_last_lr()[0], self.dis_lr_scheduler.get_last_lr()[0]))
            self.Validation_history['gen_loss'].append(epoch_val_gen_loss / len(val_loader))
            self.Validation_history['dis_loss'].append(epoch_val_dis_loss / len(val_loader))
            self.Validation_history['psnr'].append(epoch_val_psnr / len(val_loader))
            self.Validation_history['ssim'].append(epoch_val_ssim / len(val_loader))
            self.Validation_history['fid'].append(epoch_val_fid / len(val_loader))

        return epoch_val_dis_loss, epoch_val_gen_loss, epoch_val_psnr, epoch_val_ssim

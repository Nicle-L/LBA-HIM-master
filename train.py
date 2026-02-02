import argparse
import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from model.model import Image_coding
from dataset import CustomDataset



def adjust_learning_rate(optimizer, epoch, init_lr, decay_epochs=[200, 300],
                         decay_factors=[1, 0.5]):
    lr = init_lr
    for i, epoch_threshold in enumerate(decay_epochs):
        if epoch > epoch_threshold:
            lr = init_lr * decay_factors[i]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(opt):
    criterion_mse = nn.MSELoss().to(device)
    net.train()
    train_dataset = CustomDataset(path=opt.train_path)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    lamb = opt.lmbda
    for epoch in range(opt.train_epoch):
        cur_lr = adjust_learning_rate(optimizer, epoch, opt.lr)
        for step, input_data in enumerate(train_loader):
            input_data = input_data.to(device)
            mse_total = 0.0
            bpp_total = 0.0
            state_enc = torch.zeros(
                (input_data.shape[0], 2, 64, input_data.shape[2] // 2, input_data.shape[3] // 2)).to(device=device)
            state_dec = torch.zeros(
                (input_data.shape[0], 2, 64, input_data.shape[2] // 2, input_data.shape[3] // 2)).to(device=device)
            for t in range(input_data.size(1)):
                input_step = input_data[:, t:t + 1, :, :]
                num_pixels = input_step.size()[0] * input_step.size()[1] * input_step.size()[2] * input_step.size()[
                    3]
                output, xp1, xp2, next_state_enc, next_state_dec = net(input_step, state_enc, state_dec, if_training=1)

                train_bpp1 = torch.sum(torch.log(xp1)) / (-np.log(2) * num_pixels) 
                train_bpp2 = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels) 
                Bpppb =  train_bpp1 +  train_bpp2

                mse_eve = criterion_mse(output, input_step)

                l_rec = lamb * mse_eve + Bpppb

                optimizer.zero_grad()
                l_rec.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()

                mse_total += mse_eve
                bpp_total += Bpppb

                state_enc = next_state_enc.detach()
                state_dec = next_state_dec.detach()


            mse_avg = mse_total / (input_data.size(1))
            bpp_total_avg = bpp_total / (input_data.size(1))

            loss = lamb * mse_avg + bpp_total_avg
            psnr_avg = 10.0 * np.log10(1. / mse_avg.item())  
            print(f'Epoch: {epoch}, Step: {step}, MSE: {mse_avg:.6f}, Bpppb: {bpp_total_avg:.2f}, PSNR: {psnr_avg:.2f}')

            log_path = os.path.join(opt.out_dir_train, f'train_mse{int(opt.lmbda)}.log')
            with open(log_path, 'a') as fd:
                fd.write(f'ep:{epoch} step:{step} loss:{loss.item():.6f} MSE:{mse_avg:.6f} '
                         f'bpp_total:{bpp_total_avg:.4f} PSNR:{psnr_avg:.2f}\n')

        if (epoch + 1) % opt.save_interval == 0:
            test(opt)
            checkpoint_path = os.path.join(opt.checkpoint_dir, f'model_epoch_{epoch + 1}_lambda{opt.lmbda}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f'Model saved at {checkpoint_path}')


def test(opt):
    criterion_mse = nn.MSELoss().to(device)
    net.eval()
    test_dataset = CustomDataset(path=opt.test_path)
    test_loader = DataLoader(test_dataset, batch_size=opt.testBatchSize, shuffle=True, num_workers=0)
    with torch.no_grad():
        total_test_bpp = 0.0
        total_test_mse = 0.0
        total_psnr = 0.0  
        for step, input_data in enumerate(test_loader):
            input_data = input_data.to(device)
            mse_total = 0.0
            bpp_total = 0.0

            state_enc = torch.zeros(
                (input_data.shape[0], 2, 64, input_data.shape[2] // 2, input_data.shape[3] // 2)).to(device=device)
            state_dec = torch.zeros(
                (input_data.shape[0], 2, 64, input_data.shape[2] // 2, input_data.shape[3] // 2)).to(device=device)

            for t in range(input_data.size(1)):
                input_step = input_data[:, t:t + 1, :, :]
                num_pixels = input_step.size()[0] * input_step.size()[1] * input_step.size()[2] * input_step.size()[
                    3]
                output, xp1, xp2, next_state_enc, next_state_dec = net(input_step, state_enc, state_dec, if_training=0)

                train_bpp1 = torch.sum(torch.log(xp1)) / (-np.log(2) * num_pixels) 
                train_bpp2 = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels) 
                Bpppb =  train_bpp1 +  train_bpp2

                mse_eve = criterion_mse(output, input_step)

                mse_total += mse_eve
                bpp_total += Bpppb
                state_enc = next_state_enc.detach()
                state_dec = next_state_dec.detach()

            mse_avg = mse_total / (input_data.size(1))
            bpp_avg = bpp_total / (input_data.size(1))
            psnr_avg = 10.0 * np.log10(1. / mse_avg.item())
            print(f'Step: {step}, MSE: {mse_avg:.6f}, Bpppb: {bpp_avg:.2f}, PSNR: {psnr_avg:.2f}')
            total_test_bpp += bpp_avg
            total_test_mse += mse_avg
            total_psnr += psnr_avg

        total_test_bpp /= len(test_loader)
        total_test_mse /= len(test_loader)
        total_psnr /= len(test_loader)

        print(f'Average over test set - MSE: {total_test_mse:.6f}, Bpppb: {total_test_bpp:.4f}, PSNR: {total_psnr:.2f}')

        log_path = os.path.join(opt.out_dir_test, f'test{int(opt.lmbda)}.log')
        with open(log_path, 'a') as fd:
            fd.write(
                f'Average over test set - MSE: {total_test_mse:.6f}, Bpppb: {total_test_bpp:.4f}, PSNR: {total_psnr:.2f}\n')

    net.train()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='HSI Compression')
    # Model setting
    parser.add_argument('--bands', type=int, default=1)
    parser.add_argument('--M1', type=int, default=64)
    parser.add_argument('--M', type=int, default=128)
    parser.add_argument("--train_epoch", type=int, default=200)
    parser.add_argument('--train_size', type=int, default=204) #HSIMix train 204 Cave train 24
    parser.add_argument('--test_size', type=int, default=52) #HSIMix test  52  Cave test 7
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--testBatchSize', type=int, default=1)
    parser.add_argument("--lambda", type=float, default=2048, dest="lmbda", help="Lambda for rate-distortion tradeoff.")#lamda in our paper
    parser.add_argument('--out_dir_train', type=str, default='Result Path to train')
    parser.add_argument('--out_dir_test', type=str, default='Result Path to test')
    parser.add_argument('--checkpoint_dir', type=str, default='Result Path to weight')
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--train_path", type=str, default="Txt Path to train")
    parser.add_argument("--test_path", type=str, default="Txt Path to test")
    parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate.")
    opt = parser.parse_args()
    net = Image_coding(opt.bands, opt.M1, opt.M).to(device)
    print(opt)
    train(opt)

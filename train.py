import torch
from models.generator import GCCRN
from models.discriminator import Discriminator_MelGAN
import argparse
from data import PLCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from utils import *
def train(args):
    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    generator = GCCRN(2,2,257).to(device)
    discriminator = Discriminator_MelGAN().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    train_dataset = PLCDataset('/home/nick/Documents/gitproj/RealTimeBWE/datasets/vctk/train.txt', seg_len=2, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    win = torch.sqrt(torch.cat((torch.hann_window(480), torch.zeros((32,)))).to(device))
    stft_func = lambda x: torch.stft(x, n_fft=512, hop_length=160, win_length=512, window=win, return_complex=True)
    istft_func = lambda x: torch.istft(x, n_fft=512, hop_length=160, win_length=512, window=win)
    loss_func = torch.nn.MSELoss()
    exp_path = os.path.join('exp',args.version)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    for epoch in range(args.num_epoch):
        with tqdm(total=len(train_dataloader)) as t:
            for wav_input, wav_label in train_dataloader:
                wav_input, wav_label = wav_input.to(device), wav_label.to(device)
                spec_input, spec_label = stft_func(wav_input.squeeze(1)), stft_func(wav_label.squeeze(1))
                spec_infer = generator(spec_input)
                wav_infer, wav_label = istft_func(spec_infer), istft_func(spec_label)

                # update discriminator
                d_optimizer.zero_grad()
                loss_d = discriminator.loss_D(wav_infer, wav_label)
                loss_d.backward()
                d_optimizer.step()

                # update generator
                g_optimizer.zero_grad()
                loss_mse = loss_func(spec_infer.real, spec_label.real) + loss_func(spec_infer.imag, spec_label.imag)
                loss_g = loss_mse + discriminator.loss_G(wav_infer, wav_label)
                loss_g.backward()
                g_optimizer.step()

                lsd = lsd_distance((spec_label.real ** 2 + spec_label.imag ** 2).sqrt(), (spec_infer.real ** 2 + spec_infer.imag ** 2).sqrt())

                # Description will be displayed on the left
                t.set_description('Epoch %i' % epoch)
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                t.set_postfix(loss_g=loss_g.item(), loss_d=loss_d.item(), lsd=lsd.item(), mse=loss_mse.item())
                t.update()



        if epoch % args.val_inteval == 0:
            state = {'g': generator.state_dict(), 'd': discriminator.state_dict(), 'opt_g': g_optimizer.state_dict(), 'opt_d': d_optimizer.state_dict(),
                     'epoch': epoch}
            torch.save(state, '%s/epoch_%d.pth' % (exp_path, epoch))


if __name__ == "__main__":
    '''
    TODO
    1、修改训练时长为3s取2s
    2、vad
    3、enhance
    4、G-E model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=50, help='num_epoch')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--val_inteval', type=int, default=10, help='val_inteval')
    parser.add_argument('--version', '-v', type=str, required=True, help='version')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or not')
    args = parser.parse_args()
    train(args)
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


class FTLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FTLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


    def forward(self, input, hx=None):
        if hx is None:
            hx = (torch.zeros(input.size(0), self.hidden_size, device=input.device),
                  torch.zeros(input.size(0), self.hidden_size, device=input.device))


class gate_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(gate_conv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(2, 3), stride=(1, 2), padding=(1,0))

        self.bn = nn.BatchNorm2d(out_ch)
        self.elu = nn.ELU(inplace=True)
        self.pad = (0, 1)

    def forward(self, x):
        x = F.pad(x, self.pad, "constant", 0)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1[:,:,:-1,:]
        x2 = x2[:,:,:-1,:]
        x2 = torch.sigmoid(x2)
        x = x1*x2
        x = self.bn(x)
        x = self.elu(x)

        return x


class gate_deconv(nn.Module):
    def __init__(self, in_ch, out_ch, pad=0):
        super(gate_deconv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,3), stride=(1,2), output_padding=pad)
        self.deconv2 = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,3), stride=(1,2), output_padding=pad)

        self.bn = nn.BatchNorm2d(out_ch)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x2 = torch.sigmoid(x2)
        x = x1*x2
        x = self.bn(x)
        x = self.elu(x)

        return x[:,:,:,:-1]


class GCCRN(nn.Module):
    def __init__(self, in_chan, out_chan, f_dim=257):
        super(GCCRN, self).__init__()

        self.gate_conv0 = gate_conv(in_chan, 8)
        self.gate_conv1 = gate_conv(8,16)
        self.gate_conv2 = gate_conv(16,32)
        self.gate_conv3 = gate_conv(32,64)
        self.gate_conv4 = gate_conv(64,128)
        self.gate_conv5 = gate_conv(128,256)

        self.lstm1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)

        self.gate_deconv1_1 = gate_deconv(256,128)
        self.gate_deconv2_1 = gate_deconv(128,64)
        self.gate_deconv3_1 = gate_deconv(64,32)
        self.gate_deconv4_1 = gate_deconv(32,16)
        self.gate_deconv5_1 = gate_deconv(16,8)
        self.gate_deconv6_1 = gate_deconv(8, 1)

        self.gate_deconv1_2 = gate_deconv(256,128)
        self.gate_deconv2_2 = gate_deconv(128,64)
        self.gate_deconv3_2 = gate_deconv(64,32)
        self.gate_deconv4_2 = gate_deconv(32,16)
        self.gate_deconv5_2 = gate_deconv(16,8)
        self.gate_deconv6_2 = gate_deconv(8, 1)

        self.fc1 = nn.Linear(256, f_dim)
        self.fc2 = nn.Linear(256, f_dim)

        # self._initialize_weights()

    def forward(self, x_ori):
        x = torch.cat((x_ori.real.unsqueeze(2), x_ori.imag.unsqueeze(2)), dim=2).permute(0,3,2,1)
        batch, time, channel, bin = x.size()
        x = x.transpose(1,2)
        x0 = self.gate_conv0(x)
        # print('x0: ', x0.size())
        x1 = self.gate_conv1(x0)
        # print('x1: ', x1.size())
        x2 = self.gate_conv2(x1)
        # print('x2: ', x2.size())
        x3 = self.gate_conv3(x2)
        # print('x3: ', x3.size())
        x4 = self.gate_conv4(x3)
        # print('x4: ', x4.size())
        x5 = self.gate_conv5(x4)
        # print('x5: ', x5.size())
        x_lstm = x5.transpose(1,2)
        x_lstm = x_lstm.reshape([batch, time, -1])
        x_lstm_out1, (_,_) = self.lstm1(x_lstm[:,:,:512])
        x_lstm_out2, (_,_) = self.lstm2(x_lstm[:,:,512:])
        x_lstm_out = torch.cat([x_lstm_out1, x_lstm_out2], dim=-1)
        # print('x_lstm_out', x_lstm_out.size())
        x_lstm_out = x_lstm_out.view([batch, time, 256, 4])
        x_lstm_out = x_lstm_out.transpose(1,2)

        x_up1 = self.gate_deconv1_1(x_lstm_out + x5)
        # print('x_up1: ',x_up1.size())
        x_up2 = self.gate_deconv2_1(x_up1 + x4)
        # print('x_up2: ',x_up2.size())
        x_up3 = self.gate_deconv3_1(x_up2 + x3)
        # print('x_up3: ',x_up3.size())
        x_up4 = self.gate_deconv4_1(x_up3 + x2)
        # print('x_up4: ',x_up4.size())
        x_up5 = self.gate_deconv5_1(x_up4 + x1)
        # print('x_up5: ',x_up5.size())
        x_up6 = self.gate_deconv6_1(x_up5 + x0)
        # print('x_up6: ',x_up6.size())
        Cout1 = self.fc1(x_up6).transpose(2,3)
        # print('Cout1: ', Cout1.size())

        x_up1 = self.gate_deconv1_2(x_lstm_out + x5)
        # print('x_up1: ',x_up1.size())
        x_up2 = self.gate_deconv2_2(x_up1 + x4)
        # print('x_up2: ',x_up2.size())
        x_up3 = self.gate_deconv3_2(x_up2 + x3)
        # print('x_up3: ',x_up3.size())
        x_up4 = self.gate_deconv4_2(x_up3 + x2)
        # print('x_up4: ',x_up4.size())
        x_up5 = self.gate_deconv5_2(x_up4 + x1)
        # print('x_up5: ',x_up5.size())
        x_up6 = self.gate_deconv6_2(x_up5 + x0)
        # print('x_up6: ',x_up6.size())
        Cout2 = self.fc2(x_up6).transpose(2,3)


        return torch.complex(Cout1.squeeze(1),Cout2.squeeze(1)) + x_ori

    def _initialize_weights(self):
        # Xavier Init
        for name, param in self.named_parameters():
            if len(param.size()) > 1:  # 权重初始化
                init.xavier_normal_(param, gain=1)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    batch = 4
    time = 10
    channel = 2
    bin = 257
    # B, T, C, F
    data = torch.randn([batch, time, channel, bin])
    model = GCCRN(2, 2, f_dim=bin)
    out = model(data)
    print(out.size())

    result = get_parameter_number(model)
    print('Number of parameter: \n\t total: {:.2f} M, '
          'trainable: {:.2f} M'.format(result['Total'] / 1e6, result['Trainable'] / 1e6))

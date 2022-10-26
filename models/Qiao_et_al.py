import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
'''
1. LSD, FD module architecture.
'''

class LSD_FD_model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LSD_FD_model, self).__init__()

        h_f = 16

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=h_f, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(h_f),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_f, out_channels=h_f*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(h_f*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_f*4, out_channels=h_f*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(h_f*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_f*8, out_channels=h_f*16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(h_f*16),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=h_f*16, out_channels=h_f*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(h_f),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=h_f*8, out_channels=h_f*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(h_f*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_f*4, out_channels=h_f, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(h_f),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_f, out_channels=h_f, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(h_f),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=h_f, out_channels=1, kernel_size=7, stride=1, padding=3),
        )

        self._init_weight()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                m.bias.data.zero_()

'''
2. FR and FG module
'''
class Residual_block(nn.Module):
  def __init__(self, in_ch):
    super(Residual_block, self).__init__()

    self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1, padding_mode='reflect')
    self.activation = nn.ReLU(inplace=True)
    self.norm = nn.InstanceNorm2d(in_ch)

    self._init_weight()

  def forward(self, x):
    out = self.activation(self.norm(self.conv1(x)))
    out = self.norm(self.conv1(out))

    return x + out

  def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                m.bias.data.zero_()

class FR_FG_model(nn.Module):
  def __init__(self, block_num, in_ch, out_ch):
    super(FR_FG_model, self).__init__()
    
    h_f = 64
    
    self.encoder = nn.Sequential(
        nn.Conv2d(in_ch, h_f, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
        nn.InstanceNorm2d(h_f),
        nn.ReLU(),
        nn.Conv2d(h_f, 2*h_f, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(2*h_f),
        nn.ReLU(),
        nn.Conv2d(2*h_f, 4*h_f, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(4*h_f),
        nn.ReLU()
    )

    modules = [Residual_block(4*h_f) for _ in range(block_num)]
    self.ResBlock = nn.Sequential(*modules)

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(4*h_f, 2*h_f, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(h_f),
        nn.ReLU(),
        nn.ConvTranspose2d(2*h_f, h_f, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(2*h_f),
        nn.ReLU(),
        nn.Conv2d(h_f, out_ch, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
    )
  
    self._init_weight()

  def forward(self, x):
    x = self.encoder(x)
    x = self.ResBlock(x)
    x = self.decoder(x)

    return x

  def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                m.bias.data.zero_()

'''
3. Discriminator
'''

class PatchGANDiscriminator(nn.Module):
  def __init__(self, in_ch):
    super(PatchGANDiscriminator, self).__init__()
    h_f = 64

    self.activation = nn.LeakyReLU(0.2, inplace=True)
    self.conv1 = nn.Conv2d(in_ch, h_f, kernel_size=4, stride=2, padding=1)
    self.conv2 = nn.Conv2d(h_f, 2*h_f, kernel_size=4, stride=2, padding=1)
    self.norm1 = nn.InstanceNorm2d(2*h_f)
    self.conv3 = nn.Conv2d(2*h_f, 4*h_f, kernel_size=4, stride=2, padding=1)
    self.norm2 = nn.InstanceNorm2d(4*h_f)
    self.conv4 = nn.Conv2d(4*h_f, 8*h_f, kernel_size=4, stride=1, padding=1)
    self.norm3 = nn.InstanceNorm2d(8*h_f)
    self.conv5 = nn.Conv2d(8*h_f, 1, kernel_size=4, stride=1, padding=1)

    self.sigmoid = nn.Sigmoid()

    self._init_weight()

  def forward(self, x):
    x = self.activation(self.conv1(x))
    x = self.activation(self.norm1(self.conv2(x)))
    x = self.activation(self.norm2(self.conv3(x)))
    x = self.activation(self.norm3(self.conv4(x)))
    x = self.conv5(x)

    x = self.sigmoid(x)

    return x

  def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                m.bias.data.zero_()
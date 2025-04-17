# model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Paste ConvBlock class definition here ---
class ConvBlock(nn.Module):
    """Standard double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# --- Paste Attention_block class definition here ---
class Attention_block(nn.Module):
    """Attention Block"""
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# --- Paste AttU_Net class definition here ---
class AttU_Net(nn.Module):
    """Attention U-Net model"""
    def __init__(self, img_ch=3, output_ch=1): # Ensure defaults match your trained model
        super(AttU_Net, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(img_ch, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4]) # Bottleneck
        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = ConvBlock(filters[4], filters[3])
        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2])
        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])
        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])
        self.Conv_1x1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = self.Conv1(x); x2 = self.Maxpool1(x1); x2 = self.Conv2(x2)
        x3 = self.Maxpool2(x2); x3 = self.Conv3(x3); x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4); x5 = self.Maxpool4(x4); x5 = self.Conv5(x5)
        d5 = self.Up5(x5); x4_att = self.Att5(g=d5, x=x4); d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.Up_conv5(d5); d4 = self.Up4(d5); x3_att = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1); d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4); x2_att = self.Att3(g=d3, x=x2); d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3); d2 = self.Up2(d3); x1_att = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1); d2 = self.Up_conv2(d2)
        out = self.Conv_1x1(d2)
        return out
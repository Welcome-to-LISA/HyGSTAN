import torch
from torch import nn
from module import GSAM, GSTAM, SSFM


class hygstan(nn.Module):
    def __init__(self, num_patches, image_size, p=13, d=64):
        super().__init__()
        self.p = p
        self.d = d
        self.num_patches = num_patches
        self.gsam = GSAM((image_size ** 2)-p, self.num_patches, d=d)
        self.gstam = GSTAM(self.num_patches, d=d)
        self.fc = nn.Sequential(
            nn.BatchNorm1d((num_patches) * (image_size ** 2)),
            nn.LeakyReLU(inplace=True),
            nn.Linear((num_patches) * (image_size ** 2), 2, bias=True))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        ssfm_c1, ssfm_r1 = SSFM(x1, p=self.p)
        ssfm_c2, ssfm_r2 = SSFM(x2, p=self.p)
        x1 = ssfm_c1(x1)
        x2 = ssfm_c2(x2)
        x12, z1 = self.gsam(x1)
        x22, z2 = self.gsam(x2)
        src12 = self.gstam(x12, z2)
        src22 = self.gstam(x22, z1)
        x11 = ssfm_r1(src12)
        x22 = ssfm_r2(src22)
        out = self.softmax(self.fc(torch.flatten((x11 + x22).transpose(1, 2), 1, 2)))
        return out
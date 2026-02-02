import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ASC(nn.Module):
     """ Attentive Skip Connection
     """

     def __init__(self, channel):
         super().__init__()
         self.weight = nn.Sequential(
             nn.Conv3d(channel * 2, channel, 1),
             nn.LeakyReLU(),
             nn.Conv3d(channel, channel, 3, 1, 1),
             nn.Sigmoid()
         )

     def forward(self, x, y):
         x = x.unsqueeze(2)
         y = y.unsqueeze(2)
         w = self.weight(torch.cat([x, y], dim=1))
         out = (1 - w) * x + w * y
         return out.squeeze(2)



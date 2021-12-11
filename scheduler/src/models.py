import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from provisioner.src.dlutils import *
torch.manual_seed(1)

## Simple FCN Network
class GOBI(nn.Module):
	def __init__(self, feats):
		super(GOBI, self).__init__()
		self.name = 'GOBI'
		self.lr = 0.02
		self.n_feats = feats
		self.n_apps = 7
		self.fcn = nn.Sequential(nn.Linear(self.n_feats * 2 + self.n_apps, 1), nn.Sigmoid())

	def forward(self, d, a, s):
		return self.fcn(torch.cat((d.view(-1), a.view(-1), s.view(-1))))

## Simple NPN Network
class GOSH(nn.Module):
	def __init__(self, feats):
		super(GOSH, self).__init__()
		self.name = 'GOSH'
		self.lr = 0.02
		self.n_feats = feats
		self.n_apps = 7
		self.fcn = nn.Sequential( 
			NPNLinear(self.n_feats * 2 + self.n_apps, feats, False), 
			NPNRelu(),
			NPNLinear(feats, feats),
			NPNRelu(),
			NPNLinear(feats, 1),
			NPNSigmoid())

	def forward(self, d, a, s):
		x, s = self.fcn(torch.cat((d.view(-1), a.view(-1), s.view(-1))).reshape(1, -1))
		return x, s
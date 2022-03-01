import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .dlutils import *
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

## Simple Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.005
		self.n_feats = feats
		self.n_window = 1 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats * self.n_window, self.n_feats), nn.Sigmoid())

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return self.fcn(g.view(-1))

## Simple NPN based stochastic model
class NPN(nn.Module):
	def __init__(self, feats):
		super(NPN, self).__init__()
		self.name = 'NPN'
		self.lr = 0.002
		self.n_feats = feats
		self.n_window = 1 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.fcn = nn.Sequential( 
			NPNLinear(self.n, feats, False), 
			NPNRelu(),
			NPNLinear(feats, feats),
			NPNRelu(),
			NPNLinear(feats, feats),
			NPNSigmoid())

	def forward(self, g):
		x = g.reshape(1, -1)
		x, s = self.fcn(x)
		return x.view(-1), s.view(-1)

# Graph Neural Network
class GNN(nn.Module):
	def __init__(self, feats):
		super(GNN, self).__init__()
		self.name = 'GNN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 1
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats
		src_ids = np.repeat(np.array(list(range(feats))), feats)
		dst_ids = np.array(list(range(feats))*feats)
		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
		self.g = dgl.add_self_loop(self.g)
		self.feature_gat = GATConv(1, 1, feats)
		self.attention = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
		)
		self.fcn = nn.Sequential(
			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
		)

	def forward(self, data):
		# Bahdanau style attention
		att_score = self.attention(data).view(self.n_window, 1)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.matmul(data.permute(1, 0), att_score)
		# GAT convolution on complete graph
		feat_r = self.feature_gat(self.g, data_r)
		feat_r = feat_r.view(self.n_feats, self.n_feats)
		# Pass through a FCN
		x = self.fcn(feat_r)
		return x.view(-1)

## LSTM Model
class LSTM(nn.Module):
	def __init__(self, feats):
		super(LSTM, self).__init__()
		self.name = 'LSTM'
		self.lr = 0.004
		self.n_feats = feats
		self.n_window = 5
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.fcn = nn.Sequential(nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid())
		self.hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float), torch.randn(1, 1, self.n_hidden, dtype=torch.float))

	def forward(self, x):
		out, self.hidden = self.lstm(x.view(1, 1, -1), self.hidden)
		out = self.fcn(out.view(-1))
		self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
		return out
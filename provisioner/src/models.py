import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from .dlutils import *
from .constants import *
torch.manual_seed(1)

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
		return x, s

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

## Simple Transformer Model
class Transformer(nn.Module):
	def __init__(self, feats):
		super(Transformer, self).__init__()
		self.name = 'Transformer'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = 1
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

## HASCO Model
class HASCO(nn.Module):
	def __init__(self, feats):
		super(HASCO, self).__init__()
		self.name = 'HASCO'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = 1
		self.n = self.n_feats * self.n_window
		self.n_apps = 7; self.n_choices = 3
		self.q = nn.Sequential(nn.Linear(self.n_feats + self.n_apps + 2 + self.n_choices + self.n_feats, 1), nn.Sigmoid())

	def forward(self, cpu, app, prov, dec, sched):
		score = self.q(torch.cat((cpu.view(-1), app.view(-1), prov.view(-1), dec.view(-1), sched.view(-1))))
		return score

## RecSim (BNN) Model
class RecSim(nn.Module):
	def __init__(self, feats):
		super(RecSim, self).__init__()
		self.name = 'RecSim'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = 1
		self.n = self.n_feats * self.n_window
		self.n_apps = 7; self.n_choices = 3
		self.q = nn.Sequential(nn.Linear(self.n_feats + self.n_apps + 2 + self.n_choices + self.n_feats, 64), 
					BayesianLinear(64, 1),
					nn.Sigmoid())

	def forward(self, cpu, app, prov, dec, sched):
		score = self.q(torch.cat((cpu.view(-1), app.view(-1), prov.view(-1), dec.view(-1), sched.view(-1))))
		return score

## SecoNet Model
class SecoNet(nn.Module):
	def __init__(self, feats):
		super(SecoNet, self).__init__()
		self.name = 'SecoNet'
		self.lr = 0.002
		self.n_feats = feats
		self.n_window = 1
		self.n = self.n_feats * self.n_window
		self.n_apps = 7; self.n_choices = 3
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()
		self.likelihood_1 = nn.Sequential(nn.Linear(1 + feats, 2), nn.Softmax())
		self.likelihood_2 = nn.Sequential(nn.Linear(self.n_apps + feats, self.n_choices), nn.Softmax())
		self.likelihood_3 = nn.Sequential(nn.Linear(self.n_apps + feats, feats), nn.Softmax())

	def predwindow(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return memory, x

	def forward_provisioner(self, memory, hv):
		score_1 = self.likelihood_1(torch.cat((hv.view(-1), memory.view(-1))))
		return score_1

	def forward_decider(self, memory, dv):
		score_2 = self.likelihood_2(torch.cat((dv.view(-1), memory.view(-1))))
		return score_2

	def forward_scheduler(self, memory, sv):
		score_3 = self.likelihood_3(torch.cat((sv.view(-1), memory.view(-1))))
		return score_3
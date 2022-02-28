import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Simple FCN Network
class Surrogate(nn.Module):
	def __init__(self, num_hosts, feats):
		super(Surrogate, self).__init__()
		self.name = 'Surrogate'
		self.lr = 0.00005
		self.n_feats = feats
		self.n_hidden = 16
		self.fcn = nn.Sequential(
			nn.Linear(num_hosts * feats + num_hosts, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU())
		self.head1 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, 1), nn.LeakyReLU())
		self.head2 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, 1), nn.LeakyReLU())

	def forward(self, cpu, params):
		x = self.fcn(torch.cat((cpu, params), dim=1))
		e = self.head1(x)
		r = self.head2(x)
		return torch.cat((e, r), dim=1)
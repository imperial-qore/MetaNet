import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Simple FCN Network
class Surrogate(nn.Module):
	def __init__(self, num_hosts, num_schedulers):
		super(Surrogate, self).__init__()
		self.name = 'Surrogate'
		self.lr = 0.008
		self.n_hosts = num_hosts
		self.n_sched = num_schedulers
		self.n_hidden = 16
		self.fcn = nn.Sequential(
			nn.Linear(num_hosts, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_sched), nn.Sigmoid())

	def forward(self, cpu):
		x = self.fcn(cpu.view(-1))
		return x
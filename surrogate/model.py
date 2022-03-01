import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Simple FCN Network
class Surrogate(nn.Module):
	def __init__(self, num_hosts, num_schedulers):
		super(Surrogate, self).__init__()
		self.name = 'Surrogate'
		self.lr = 0.005
		self.n_hosts = num_hosts
		self.n_sched = num_schedulers
		self.n_hidden = 16
		self.fcn = nn.Sequential(
			nn.Linear(num_hosts, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU())
		self.head1 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, num_schedulers), nn.Sigmoid())
		self.head2 = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(),
            nn.Linear(self.n_hidden, num_schedulers), nn.Sigmoid())

	def forward(self, cpu):
		x = self.fcn(cpu.view(-1))
		r = self.head1(x)
		s = self.head2(x)
		return r, s
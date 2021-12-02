from .Provisioner import *
import random

class RandomProvisioner(Provisioner):
	def __init__(self):
		super().__init__()

	def provision(self):
		for host in self.env.hostlist:
			todo = random.choice([0, 1])
			if todo == 1:
				host.enable = True
			else:
				host.enable = False
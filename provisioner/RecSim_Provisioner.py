from .HASCO_Provisioner import *
from provisioner.src.utils import *

class RecSimProvisioner(HASCOProvisioner):
	def __init__(self):
		super().__init__()
		self.model_name = 'RecSim'

	def provision(self):
		if not self.model_loaded: self.load_model()
		self.host_util = torch.FloatTensor([h.getCPU() for h in self.env.hostlist]) / 100
		cpu = self.host_util
		for i in range(len(self.env.hostlist)):
			scores = [self.testall(cpu, prov=i) for i in [0, 1]]
			todo = np.argmax(scores)
			self.env.hostlist[i].enable = True if todo == 1 else False
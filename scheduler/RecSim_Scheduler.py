from .HASCO_Scheduler import *
from provisioner.src.utils import *
from .src.opt import *

class RecSimScheduler(HASCOScheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'RecSim'
		
	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		cpu = self.host_util
		decisions = []
		for task in tasks:
			dec = np.argmax([self.testall(cpu, prov=1, app=task.application, dec=task.choice, sched=i) for i in list(range(len(self.env.hostlist)))])
			decisions.append(dec)
		return decisions, time() - start
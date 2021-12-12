from .Scheduler import *
from provisioner.src.utils import *
from .src.opt import *

class HASCOScheduler(Scheduler):
	def __init__(self):
		super().__init__()
		self.model_name = 'HASCO'
		self.search = LocalSearch
		self.model_loaded = False
		self.fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]

	def load_model(self):
		# Assume pretrained model loaded by provisioner
		self.feats = len(self.env.hostlist)
		self.model = self.env.provisioner.model
		self.testall = self.env.provisioner.testall
		self.host_util = self.env.provisioner.host_util
		self.model_loaded = True

	def placement(self, tasks):
		if not self.model_loaded: self.load_model()
		start = time()
		cpu = self.host_util
		decisions = []
		for task in tasks:
			dec = np.argmax([self.testall(cpu, prov=1, app=task.application, dec=task.choice, sched=i) for i in list(range(len(self.env.hostlist)))])
			decisions.append(dec)
		return decisions, time() - start
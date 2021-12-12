from .Decider import *
from provisioner.src.utils import *
from .src.opt import *

class HASCODecider(Decider):
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

	def decision(self, workflowlist):
		if not self.model_loaded: self.load_model()
		cpu = self.host_util
		results = []
		for i, (CreationID, interval, SLA, application) in enumerate(workflowlist):
			choice = self.choices[np.argmax([self.testall(cpu, app=application, dec=i) for i in self.choices])]
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
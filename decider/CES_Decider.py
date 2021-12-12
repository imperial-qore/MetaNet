from .Decider import *
from provisioner.src.utils import *
from .src.opt import *

class CESDecider(Decider):
	def __init__(self):
		super().__init__()
		self.model_name = 'CES'
		self.search = LocalSearch
		self.model_loaded = False

	def load_model(self):
		# Assume pretrained model loaded by provisioner
		self.feats = len(self.env.hostlist)
		self.model = self.env.provisioner.model
		self.host_util = self.env.provisioner.host_util
		self.model_loaded = True

	def decision(self, workflowlist):
		if not self.model_loaded: self.load_model()
		predips = self.env.provisioner.predips
		opt = self.search(predips, self.env, [w[3] for w in workflowlist], [w[2] for w in workflowlist])
		decisions = opt.search()
		results = []
		for i, (CreationID, interval, SLA, application) in enumerate(workflowlist):
			choice = decisions[i]
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
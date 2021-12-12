from .HASCO_Decider import *
from provisioner.src.utils import *
from .src.opt import *

class RecSimDecider(HASCODecider):
	def __init__(self):
		super().__init__()
		self.model_name = 'RecSim'

	def decision(self, workflowlist):
		if not self.model_loaded: self.load_model()
		cpu = self.host_util
		results = []
		for i, (CreationID, interval, SLA, application) in enumerate(workflowlist):
			choice = self.choices[np.argmax([self.testall(cpu, app=application, dec=i) for i in self.choices])]
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
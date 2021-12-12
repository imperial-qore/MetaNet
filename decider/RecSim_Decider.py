from .HASCO_Decider import *
from provisioner.src.utils import *
from .src.utils import *
from .src.opt import *

class RecSimDecider(HASCODecider):
	def __init__(self):
		super().__init__()
		self.model_name = 'RecSim'
		self.rt_dict, self.sla_dict = get_dicts()
		self.a_dict = get_accuracies()
		self.xi = 1 # weight of accuracy w.r.t SLA violation 

	def getSimScores(self, application, sla):
		sla = self.sla_dict[application]
		rt_s = [np.random.normal(loc = self.rt_dict[application][choice][0], scale = self.rt_dict[application][choice][1]) for choice in self.choices]
		sla_s = [rt <= sla for rt in rt_s]
		a_s = [self.a_dict[application][choice] for choice in self.choices]
		choice_scores = [sla_s[i] + self.xi * a_s[i] for i in range(len(self.choices))]
		return choice_scores

	def decision(self, workflowlist):
		if not self.model_loaded: self.load_model()
		cpu = self.host_util
		results = []
		for i, (CreationID, interval, SLA, application) in enumerate(workflowlist):
			# generate BNN based recommendation scores
			scores = np.array([self.testall(cpu, app=application, dec=i) for i in self.choices])
			# generate simulator based recommendation scores
			sim_scores = self.getSimScores(application, SLA)
			# run collaborative filtering
			scores += sim_scores
			choice = self.choices[np.argmax(scores)]
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
from .Decider import *
from .src.utils import *
from pprint import pprint

class CoSimDecider(Decider):
	def __init__(self):
		super().__init__()
		self.rt_dict, self.sla_dict = get_dicts()
		self.a_dict = get_accuracies()
		self.xi = 1 # weight of accuracy w.r.t SLA violation 
	
	def getChoice(self, application, sla):
		sla = self.sla_dict[application]
		rt_s = [np.random.normal(loc = self.rt_dict[application][choice][0], scale = self.rt_dict[application][choice][1]) for choice in self.choices]
		sla_s = [rt <= sla for rt in rt_s]
		a_s = [self.a_dict[application][choice] for choice in self.choices]
		choice_scores = [sla_s[i] + self.xi * a_s[i] for i in range(len(self.choices))]
		# print(sla_s, a_s)
		return self.choices[np.argmax(choice_scores)]

	def decision(self, workflowlist):
		results = []
		for CreationID, interval, SLA, application in workflowlist:
			choice = self.getChoice(application, SLA)
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
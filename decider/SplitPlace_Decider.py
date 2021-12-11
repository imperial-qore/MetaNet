from .Decider import *
from .src.utils import *

class SplitPlaceDecider(Decider):
	def __init__(self):
		super().__init__()
		self.rt_dict, self.sla_dict = get_dicts()
		self.a_dict = get_accuracies()
		self.xi = 1 # weight of accuracy w.r.t SLA violation 
	
	def getChoice(self, application, sla):
		sla = self.sla_dict[application]
		# decision between only layer and semantic
		rt_s = np.random.normal(loc = self.rt_dict[application]['layer'][0], scale = self.rt_dict[application]['layer'][1])
		if sla < rt_s:
			return 'semantic'
		return 'layer'

	def decision(self, workflowlist):
		results = []
		for CreationID, interval, SLA, application in workflowlist:
			choice = self.getChoice(application, SLA)
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
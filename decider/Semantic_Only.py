from .Decider import *

class SemanticOnlyDecider(Decider):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		results = []
		for CreationID, interval, SLA, application in workflowlist:
			choice = self.choices[1]
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
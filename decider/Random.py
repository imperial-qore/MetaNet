from .Decider import *
import numpy as np

class RandomDecider(Decider):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		results = []
		for CreationID, interval, SLA, application in workflowlist:
			choice = np.random.choice(self.choices)
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
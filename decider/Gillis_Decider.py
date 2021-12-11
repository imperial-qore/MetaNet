from .Decider import *
import numpy as np

class GillisDecider(Decider):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		results = [] # only layer choice
		for CreationID, interval, SLA, application in workflowlist:
			choice = np.random.choice(self.choices, p=[0.3, 0.3, 0.4])
			tasklist = self.createTasks(CreationID, interval, SLA, application, choice)
			results += tasklist
		return results
from .Decider import *
import random

class RandomDecider(Decider):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		return random.choices(self.choices, k = len(workflowlist))
from .Decider import *

class LayerOnlyDecider(Decider):
	def __init__(self):
		super().__init__()

	def decision(self, workflowlist):
		return [self.choices[0]] * len(workflowlist)
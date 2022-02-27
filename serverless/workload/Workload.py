import numpy as np

class Workload():
	def __init__(self):
		self.creation_id = 0
		self.env = None

	def setEnvironment(self, env):
		self.env = env
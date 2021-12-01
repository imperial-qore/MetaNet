from .Scheduler import *
import numpy as np
from copy import deepcopy

class RandomScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def placement(self, tasks):
		start = time()
		return self.RandomPlacement(tasks), time() - start
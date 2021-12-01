import math
from utils.MathUtils import *
from utils.MathConstants import *
import pandas as pd
from statistics import median
import numpy as np
from time import time

class Scheduler():
    def __init__(self):
        self.env = None

    def setEnvironment(self, env):
        self.env = env

    def selection(self):
        pass

    def placement(self, containerlist):
        pass

    def filter_placement(self, decision):
        return decision

    # Task placement

    def RandomPlacement(self, tasks):
        decision = []
        for task in tasks:
            decision.append(np.random.randint(0, len(self.env.hostlist)))
        return decision

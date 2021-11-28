import math
from utils.MathUtils import *
from utils.MathConstants import *
import pandas as pd
from statistics import median
import numpy as np

class Decider():
    def __init__(self):
        self.env = None
        self.choices = ['layer', 'semantic', 'compression']

    def setEnvironment(self, env):
        self.env = env

    def decision(self, workflowlist):
        pass
        
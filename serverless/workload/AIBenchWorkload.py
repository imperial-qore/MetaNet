from .Workload import *
from utils.Utils import *
from random import gauss, choices
from serverless.task.Task import *

class AIBenchWorkload(Workload):
    def __init__(self, num_workloads, std_dev):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.dataset = list(filter(lambda k: '.md' not in k, os.listdir(SAMPLE_PATH)))
        
    def generateNewContainers(self, interval):
        workloadlist = []
        applications = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
        multiplier = np.array([2, 2, 2, 2, 2, 2, 2])
        weights = 1 - (multiplier / np.sum(multiplier))
        for i in range(max(1,int(gauss(self.num_workloads, self.std_dev)))):
            CreationID = self.creation_id
            SLA = np.random.randint(5,8) ## Update this based on intervals taken
            application = choices(applications, weights=weights)[0]
            workloadlist.append(Task(CreationID, interval, SLA, application, 'compression', self.env, 0, [], self.dataset))
            self.creation_id += 1
        return workloadlist
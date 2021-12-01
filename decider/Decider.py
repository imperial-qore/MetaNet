from serverless.task.Task import *

class Decider():
    def __init__(self):
        self.env = None
        self.choices = ['layer', 'semantic', 'compression']
        self.dataset = list(filter(lambda k: '.md' not in k, os.listdir(SAMPLE_PATH)))
        self.dataset = [os.path.join(SAMPLE_PATH, i) for i in self.dataset]

    def setEnvironment(self, env):
        self.env = env

    def decision(self, workflowlist):
        pass

    def getLayerInputs(self, cid, app, i):
        paths = []
        for d in DSET:
            d2 = d.split('.')[0]
            paths.append(f'./temp/{cid}_{i}_{app}_{d2}_L.jpg')
        return paths

    def createTasks(self, cid, interval, SLA, app, choice):
        tasklist = []
        if choice == 'semantic':
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 0, [], self.dataset))
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 1, [], self.dataset))
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 2, [], self.dataset))
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 3, [], self.dataset))
        elif choice == 'layer':
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 0, [], self.dataset))
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 1, [0], self.getLayerInputs(cid, app, 0)))
        elif choice == 'compression':
            tasklist.append(Task(cid, interval, SLA, app, choice, self.env, 0, [], self.dataset))
        return tasklist


        
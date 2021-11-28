import pandas as pd
import numpy as np
import random

class Provisioner():
    def __init__(self, datacenter, CONTAINERS):
        self.env = None
        self.datacenter = datacenter
        self.containers = CONTAINERS
        self.decision = {'remove': [], 'add': []}

    def setEnvironment(self, env):
        self.env = env

    def prediction(self):
        pass

    def provision(self):
        orphaned = []
        return self.decision, orphaned

    def removeHost(self, hostID):
        if len(self.env.hostlist) <= 1:
            return
        orphaned = self.env.getContainersOfHost(hostID)
        del self.env.hostlist[hostID]
        for cid, container in enumerate(self.env.containerlist):
            if container:
                existinghost = self.env.getContainerByID(cid).hostid
                if existinghost >= hostID:
                    newhost = len(self.env.hostlist) - 1 if existinghost == 0 else existinghost - 1
                    self.env.getContainerByID(cid).hostid = newhost
        for hid in range(len(self.env.hostlist)):
            self.env.getHostByID(hid).ID = hid
        return orphaned

    def addHost(self, hostCharacteristics):
        IPS, RAM, Disk, Bw, Latency, Powermodel = hostCharacteristics
        self.env.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)

    def migrateOrphaned(self, orphaned):
        indices = list(range(len(self.env.hostlist)))
        for o in orphaned:
            random.shuffle(indices)
            for i in indices:
                if self.env.getPlacementPossible(o, i):
                    self.env.getContainerByID(o).allocate(i, self.env.getHostByID(i).bwCap.downlink)
                    break

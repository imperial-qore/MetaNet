from simulator.host.Host import *
from simulator.container.Container import *

class Simulator():
	# Total Router Bw
	# Interval Time in seconds
	def __init__(self, RouterBw, Scheduler, Workload, ContainerLimit, IntervalTime, hostinit):
		self.totalbw = RouterBw
		self.hostlimit = len(hostinit)
		self.scheduler = Scheduler
		self.scheduler.setEnvironment(self)
		self.workload = Workload
		self.workload.setEnvironment(self)
		self.containerlimit = ContainerLimit
		self.hostlist = []
		self.waitinglist = []
		self.activetasklist = []
		self.intervaltime = IntervalTime
		self.interval = 0
		self.completedtasklist = []
		self.stats = None
		self.addHostlistInit(hostinit)

	def addHostlistInit(self, hostList):
		assert len(hostList) == self.hostlimit
		for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
			host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Latency, Powermodel, self)
			self.hostlist.append(host)

	def getContainersOfHost(self, hostID):
		containers = []
		for container in self.activetasklist:
			if container and container.hostid == hostID:
				containers.append(container.taskID)
		return containers

	def getContainerByID(self, containerID):
		return self.activetasklist[containerID]

	def getContainerByCID(self, creationID):
		for c in self.activetasklist + self.completedtasklist:
			if c and c.creationID == creationID:
				return c

	def getHostByID(self, hostID):
		return self.hostlist[hostID]

	def getPlacementPossible(self, container, hostID):
		host = self.hostlist[hostID]
		ipsreq = container.getBaseIPS()
		ramsizereq, ramreadreq, ramwritereq = container.getRAM()
		disksizereq, diskreadreq, diskwritereq = container.getDisk()
		ipsavailable = host.getIPSAvailable()
		ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
		disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
		return (ipsreq <= ipsavailable and \
				ramsizereq <= ramsizeav and \
				disksizereq <= disksizeav)

	def allocateInit(self, containers, decision):
		migrations = []; deployed = 0
		routerBwToEach = self.totalbw / len(decision)
		for i, hid in enumerate(decision):
			container = containers[i]
			assert container.getHostID() == -1
			allocbw = min(self.getHostByID(hid).bwCap.downlink, routerBwToEach)
			if self.getPlacementPossible(container, hid):
				container.allocateAndExecute(hid, allocbw)
				deployed += 1
				self.activetasklist.append(container)
		return deployed

	def destroyCompletedTasks(self):
		destroyed = []
		for i, container in enumerate(self.activetasklist):
			if container and container.getBaseIPS() == 0:
				container.destroy()
				self.activetasklist.remove(container)
				self.completedtasklist.append(container)
				destroyed.append(container)
		return destroyed
	
	def getTasksofHost(self, hostID):
		tasks = []
		for task in self.activetasklist:
			if task.hostid == hostID:
				tasks.append(task.creationID)
		return tasks
	
	def getTasksInHosts(self):
		return [len(self.getTasksofHost(host)) for host in range(len(self.hostlist))]

	def getNumActiveTasks(self):
		return len(self.activetasklist)

	def getSelectableContainers(self):
		selectable = []
		for container in self.activetasklist:
			if container and container.active and container.getHostID() != -1:
				selectable.append(container.taskID)
		return selectable

	def addContainers(self, newactivetasklist):
		self.interval += 1
		destroyed = self.destroyCompletedContainers()
		deployed = self.addactivetasklist(newactivetasklist)
		return deployed, destroyed

	def getActiveactivetasklist(self):
		return [c.getHostID() if c and c.active else -1 for c in self.activetasklist]

	def getContainersInHosts(self):
		return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

	def simulationStep(self, containers, decision):
		routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
		migrations = []; deployed = 0
		containerIDsAllocated = []
		for i, hid in enumerate(decision):
			container = containers[i]
			currentHostID = container.getHostID()
			currentHost = self.getHostByID(currentHostID)
			targetHost = self.getHostByID(hid)
			allocbw = min(targetHost.bwCap.downlink, currentHost.bwCap.uplink, routerBwToEach)
			if hid != container.hostid and self.getPlacementPossible(container, hid):
				deployed += 1
				container.allocateAndExecute(hid, allocbw)
				containerIDsAllocated.append(container.creationID)
				self.activetasklist.append(container)
		for i,container in enumerate(self.activetasklist):
			if container and container.creationID not in containerIDsAllocated:
				container.execute(0)
		return deployed
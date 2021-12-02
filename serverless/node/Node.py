from metrics.Disk import *
from metrics.RAM import *
from metrics.Bandwidth import *
from serverless.datacenter.server.controller import *

class Node():
	# IPS = Million Instructions per second capacity 
	# RAM = Ram in MB capacity
	# Disk = Disk characteristics capacity
	# Bw = Bandwidth characteristics capacity
	def __init__(self, ID, IP, IPS, RAM_, Disk_, Bw, Powermodel, Serverless):
		self.id = ID
		self.ip = IP
		self.ipsCap = IPS
		self.ramCap = RAM_
		self.diskCap = Disk_
		self.bwCap = Bw
		# Initialize utilization metrics
		self.ips = 0
		self.cpu = 0
		self.ram = RAM(0, 0, 0)
		self.bw = Bandwidth(0, 0)
		self.disk = Disk(0, 0, 0)
		self.powermodel = Powermodel
		self.powermodel.allocHost(self)
		self.powermodel.host = self
		self.env = Serverless
		
	def getPower(self):
		return self.powermodel.power()

	def getPowerFromIPS(self, ips):
		return self.powermodel.powerFromCPU(self.getCPU())

	def getCPU(self):
		# 0 - 100 last interval
		return self.cpu

	def getBaseIPS(self):
		return self.ips

	def getApparentIPS(self):
		return self.ips

	def getIPSAvailable(self):
		return self.ipsCap - self.ips

	def getCurrentRAM(self):
		return self.ram.size, self.ram.read, self.ram.write

	def getRAMAvailable(self):
		size, read, write = self.getCurrentRAM()
		return max(0, (0.6 if self.ramCap.size < 4000 else 0.8) * self.ramCap.size - size), self.ramCap.read - read, self.ramCap.write - write

	def getCurrentDisk(self):
		return self.disk.size, self.disk.read, self.disk.write

	def getDiskAvailable(self):
		size, read, write = self.getCurrentDisk()
		return self.diskCap.size - size, self.diskCap.read - read, self.diskCap.write - write

	def updateUtilizationMetrics(self):
		host_data = gethostStat(self.ip)
		print(host_data)
		self.ips = host_data['cpu'] * self.ipsCap / 100
		self.cpu = host_data['cpu']
		self.ram.size = host_data['memory']
		self.disk.size = host_data['disk']
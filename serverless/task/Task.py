from utils.Utils import *
from serverless.datacenter.server.controller import *

class Task():
	def __init__(self, creationID, creationInterval, sla, application, choice, Framework, taskID, precedence, input_imgs, HostID = -1):
		self.creationID = creationID
		self.taskID = taskID
		self.precedence = precedence
		self.choice = choice
		self.sla = sla
		self.env = Framework
		self.createAt = creationInterval
		self.application = application
		self.input_imgs = input_imgs
		self.hostid = HostID
		self.startAt = -1
		self.destroyAt = -1
		self.getOutputImgs()
		
	def getHostID(self):
		return self.hostid

	def getOutputImgs(self):
		ch = self.choice[0].upper()
		self.output_imgs = []
		for img in self.input_imgs:
			for d in DSET:
				d2 = d.split('.')[0]
				if d2 in img:
					self.output_imgs.append(f'./temp/{self.creationID}_{self.taskID}_{self.application}_{d2}_{ch}.jpg')
					break
		
	def runTask(self, ip):
		runFunctionsAll(ip, self.application, self.input_imgs, self.output_imgs)

	def destroy(self):
		self.destroyAt = self.env.interval
		self.hostid = -1
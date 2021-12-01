from .server.controller import *
from metrics.powermodels.PMRaspberryPi import *
from metrics.powermodels.PMB2s import *
from metrics.powermodels.PMB2ms import *
from metrics.powermodels.PME2asv4 import *
from metrics.powermodels.PME4asv4 import *
from metrics.powermodels.PMB4ms import *
from metrics.powermodels.PMB8ms import *
from metrics.powermodels.PMXeon_X5570 import *
from metrics.Disk import *
from metrics.RAM import *
from metrics.Bandwidth import *
from utils.Utils import *

import multiprocessing, os, platform
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

class Datacenter():
    def __init__(self, mode):
        unixify(['serverless/datacenter/agent/'])
        self.mode = mode
        self.fn_names = [name for name in os.listdir(FN_PATH) if os.path.isdir(FN_PATH+name)]
        self.dataset = list(filter(lambda k: '.md' not in k, os.listdir(SAMPLE_PATH)))
        self.dataset = [os.path.join(SAMPLE_PATH, i) for i in self.dataset]
        if os.path.exists('./temp/'): shutil.rmtree('./temp/', ignore_errors=True)
        os.makedirs('./temp/', exist_ok = True)
        if self.mode in [0, 1]:
            self.setupHosts()
        else:
            self.checkHosts()

    def cleanup(self):
        if self.mode in [0, 3]:
            self.destroyHosts()

    def setupHosts(self):
        pass

    def destroyHosts(self):
        pass

    def checkHosts(self):
        if not os.path.isfile(IPS_PATH):
            raise Exception('ips.json file does not exist')
        with open(IPS_PATH, 'r') as f:
            config = json.load(f)
        self.servers = config['servers']
        for server in self.servers:
            fn = self.fn_names[0]
            ip = server['ip']
            res = runFunctions(server['ip'], fn, self.dataset[0], 'test.jpg')
            if not res:
                raise Exception(f'Function {fn} failed on host {ip}')
        os.remove("test.jpg")

    def generateHosts(self):
        hosts = []
        print(color.HEADER+"Obtaining host information and generating Host objects"+color.ENDC)
        ips = [server['ip'] for server in self.servers]
        outputHostsData = Parallel(n_jobs=num_cores)(delayed(gethostDetails)(i) for i in ips)
        for i, data in enumerate(outputHostsData):
            IP = self.servers[i]['ip']
            print(color.BOLD+"Host details collected from: {}".format(IP)+color.ENDC, data)
            IPS = (20167518615 * self.servers[i]['cpu'])/(float(data['clock']) * 1000000)
            Power = eval(self.servers[i]['powermodel']+"()")
            Ram = RAM(data['Total_Memory'], data['Ram_read'], data['Ram_write'])
            Disk_ = Disk(data['Total_Disk'], data['Disk_read'], data['Disk_write'])
            Bw = Bandwidth(data['Bandwidth'], data['Bandwidth'])
            hosts.append((IP, IPS, Ram, Disk_, Bw, Power))
        return hosts

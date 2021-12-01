import json
import psutil
import time
import subprocess
import os
import sys

def parse_io(line):
    val = float(line.split(" ")[-2])
    unit = line.split(" ")[-1]
    if 'G' in unit: val *= 1000
    elif 'K' in unit: val /= 1000
    return val * 1.048576

def hostDetails():
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    data = subprocess.run("~/agent/probe.sh", shell=True,stdout=subprocess.PIPE)
    data  = (data.stdout.decode()).splitlines()
    interface = 'eth0'
    bw = ((subprocess.run("sudo ethtool "+interface+" | grep Speed",shell=True,stdout=subprocess.PIPE)).stdout.decode()).split()[1][0:4]
    payload ={
            "Total_Memory": int(float(memory.total/(1024*1024))),
            "Total_Disk": int(float(disk.total/(1024*1024))),
            "Bandwidth": int(bw),
            "clock": data[0],
            "Ram_read": parse_io(data[3]),
            "Ram_write": parse_io(data[4]),
            "Disk_read": parse_io(data[1]),
            "Disk_write": parse_io(data[2])}
    return json.dumps(payload)

def gethostStat():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()[2]
    disk = psutil.disk_usage('/')
    disk_total = disk.used / (1024 * 1024)
    ts = time.time()
    payload = {"time-stamp":ts, "cpu":cpu, "memory":memory, "disk":disk_total}
    return json.dumps(payload)

if __name__ == '__main__':
    if sys.argv[-1] == 'stats':
        print(gethostStat())
    else:
        print(hostDetails())
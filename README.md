# SECO
Serverless Co-Design Framework.


   <a href="https://gitpod.io/#https://github.com/shreshthtuli/SECO/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>

System decision making to meet system level objectives by exploiting the synergism of hardware and software through their concurrent design. 

## Quick Start Guide

### Installation.

```console
# install prerequisites
sudo apt -y update && apt install -y rsync
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
export PATH=$PATH:~/.local/bin
sudo chmod 400 keys/id_rsa

# install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

Or use pre-installed gitpod.

### Quick test.

```console
# deploy a single VM and print VM IP
python3 debug/deployazure.py

# test function
http http://<public_ip>:7071/api/onnx @debug/babyyoda.jpg > output.jpg
```

# Details and motivation

For cosim provisioner: tradeoff between utilization ratio and cost, decider: sla ciolation and accuracy, scheduler: qos.

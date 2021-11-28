# SECO
Serverless Co-Design Framework.
   <a href="https://gitpod.io/#https://github.com/shreshthtuli/SECO/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>


## Quick Start Guide

### Installation.

```console
# install prerequisites
python3 -m pip --upgrade
python3 -m pip install -r requirements.txt
sudo chmod 400 keys/id_rsa

# install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# update path
export PATH=$PATH:~/.local/bin
```

Or use pre-installed gitpod.

### Quick test.

```console
# deploy a single VM and print VM IP
python3 debug/deployazure.py

# test function
http http://<public_ip>:7071/api/onnx @debug/babyyoda.jpg > output.jpg
```
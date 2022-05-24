[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/MetaNet/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FMetaNet&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<a href="https://gitpod.io/#https://github.com/imperial-qore/MetaNet/">
    <img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in gitpod">
  </a>
  
# MetaNet

The operational cost of a cloud computing platform is one of the most significant Quality of Service (QoS) criteria for schedulers, crucial to keep up with the growing computational demands. Several data-driven deep neural network (DNN)-based schedulers have been proposed in recent years that outperform alternative approaches by providing scalable and effective resource management for dynamic workloads. However, state-of-the-art schedulers rely on advanced DNNs with high computational requirements, implying high scheduling costs. In non-stationary contexts, the most sophisticated schedulers may not always be required, and it may be sufficient to rely on low-cost schedulers to temporarily save operational costs. In this work, we propose MetaNet, a surrogate model that predicts the operational costs and scheduling overheads of a large number of DNN-based schedulers and chooses one on-the-fly to jointly optimize job scheduling and execution costs. This facilitates improvements in execution costs, energy usage and service level agreement violations of up to 11\%, 43\% and 13\% compared to the state-of-the-art methods.

## Quick Start Guide

### Installation.

Use the below installation instructions or spin a Gitpod container with pre-installed dependencies.

```console
# install prerequisites
sudo apt -y update && sudo apt install -y rsync python3-pip
pip3 install --upgrade pip
pip3 install matplotlib scikit-learn
pip3 install -r requirements.txt
export PATH=$PATH:~/.local/bin
sudo chmod 400 keys/id_rsa

# install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Running the MetaNet model
```console
python3 metanet.py
```

## Arxiv Link

A preliminary version of this work was accepted as a poster in ACM SIGMETRICS 2022: https://arxiv.org/pdf/2205.10640.pdf.

The conference version is available here: https://arxiv.org/pdf/2205.10642.pdf.

## Cite this work
Our work is published in IEEE CLOUD Conference. Cite using the following bibtex entry.
```bibtex
@article{tuli2022splitplace,
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R.},
  journal={IEEE CLOUD}, 
  title={{MetaNet: Automated Dynamic Selection of Scheduling Policies in Cloud Environments}}, 
  year={2022}
}
```


## License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli.
All rights reserved.

See License file for more details.




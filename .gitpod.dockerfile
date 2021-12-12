FROM gitpod/workspace-full

RUN sudo apt-get update  && sudo apt-get install -y rsync  && sudo rm -rf /var/lib/apt/lists/* && curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir matplotlib scikit-learn \
    && python3 -m pip install --no-cache-dir torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

FROM tensorflow/tensorflow:latest-gpu

RUN apt update
RUN yes | apt install git
RUN python -m pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install --upgrade requests
RUN python -m pip install --upgrade pip
RUN python -m pip install pip-tools

ARG USERNAME=henry
ENV HOME /home/${USERNAME}

COPY requirements.txt .
# COPY pyproject.toml pyproject.toml 
# RUN python -m piptools compile -o requirements.txt pyproject.toml
RUN python -m pip install -r requirements.txt

# COPY ${HOME}/.bashrc . 
CMD /bin/bash

# export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
# echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
# sudo apt-get update
# sudo apt-get install gcsfuse

# This probably needs to be done interactively
# gcloud auth application-default login

# mount the ml-checkpoints bucket 
# gcsfuse ml-checkpoints /home/henry/ckpt


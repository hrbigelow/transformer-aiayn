FROM tensorflow/tensorflow:latest-gpu

RUN apt update
RUN yes | apt install git
RUN python -m pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install --upgrade requests
RUN python -m pip install --upgrade pip
RUN python -m pip install pip-tools

# ARG USERNAME=henry
# ENV HOME /home/${USERNAME}

COPY requirements.txt .
# COPY pyproject.toml pyproject.toml 
# RUN python -m piptools compile -o requirements.txt pyproject.toml
RUN python -m pip install -r requirements.txt

# COPY ${HOME}/.bashrc . 
CMD /bin/bash

RUN GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s) && \
      echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
      | tee /etc/apt/sources.list.d/gcsfuse.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05

# RUN CLOUD_SDK_REPO=cloud-sdk-$(lsb_release -c -s) && \
  #      echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" \
  #    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

RUN apt update
RUN apt install -y gcsfuse
# RUN apt install google-cloud-sdk
WORKDIR /root

ADD https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-435.0.1-linux-x86_64.tar.gz \
  google-cloud-cli.tar.gz

RUN tar -xf google-cloud-cli.tar.gz
RUN yes | google-cloud-sdk/install.sh
# RUN /root/google-cloud-sdk/bin/gcloud init

# This probably needs to be done interactively
# gcloud auth application-default login

# mount the ml-checkpoints bucket 
# gcsfuse ml-checkpoints /home/henry/ckpt


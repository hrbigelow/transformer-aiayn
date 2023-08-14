FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && \
    apt-get install google-cloud-sdk -y

# RUN apt-get -y update
RUN apt-get -y install git

# COPY pyproject.toml pyproject.toml 
# RUN python -m piptools compile -o requirements.txt pyproject.toml
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt

# RUN groupadd -r henry && useradd -r -g henry henry

RUN python -m pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install --upgrade requests
RUN python -m pip install --upgrade pip
RUN python -m pip install pip-tools

# USER henry:henry
# ARG USERNAME=henry
# ENV HOME /home/${USERNAME}


# COPY ${HOME}/.bashrc . 

WORKDIR /root

# no idea why this works.  .rcfile is empty, yet including this instead of
# CMD ["/bin/bash"] allows the running container access to environment variables
# passed via 'docker run -e ENV'
COPY .rcfile .
CMD ["/bin/bash", "--rcfile", ".rcfile"]
# CMD /bin/bash -c HISTSIZE=${HISTSIZE} 
# RUN /root/google-cloud-sdk/bin/gcloud init

# This probably needs to be done interactively
# gcloud auth application-default login
# /root/google-cloud-sdk/bin/gcloud auth application-default set-quota-project $PROJECT_ID

# mount the ml-checkpoints bucket 
# gcsfuse ml-checkpoints /home/henry/ckpt


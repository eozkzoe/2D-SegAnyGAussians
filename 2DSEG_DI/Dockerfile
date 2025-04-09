# SAGA Pipeline Dockerfile

# Use official CUDA Docker image for v11.6
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

WORKDIR /root

# Workaround for freeze bug with tzdata installation
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update && \
    apt install -y tzdata

# Install system dependencies (includes Python 3.8)
RUN apt-get update \
    && apt-get install -y build-essential wget ninja-build unzip libgl-dev ffmpeg \
    && apt-get clean \
    && apt install -y wget git nano vim python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_HOME /opt/conda
ENV PATH $CONDA_HOME/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $CONDA_HOME && \
    rm miniconda.sh && \
    $CONDA_HOME/bin/conda clean -tipy


# Create new Conda env
# Activate Conda env with every new shell session
RUN conda init bash
# SHELL ["conda", "run", "-n", "gaussian_splatting", "/bin/bash", "-c"]
RUN $CONDA_HOME/bin/conda create -n 2dseg python=3.10 && echo "source activate 2dseg" > ~/.bashrc
ENV DISPLAY=:0

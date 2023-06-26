FROM nvidia/cuda:11.5.1-devel-ubuntu20.04

MAINTAINER robertolopezcastro <roberto.lopez.castro@udc.es>

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir projects
WORKDIR projects

# Install Dependencies
RUN apt-get -y update --fix-missing
RUN apt-get install -y git emacs wget libgoogle-glog-dev
RUN apt-get install -y software-properties-common
RUN apt-get install -y liblapack-dev
RUN apt-get install -y libblas-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libboost-dev
RUN apt install -y libgoogle-glog-dev
RUN apt-get update
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y cmake

# Install cuSparseLt
RUN wget  https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
RUN tar -xf libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
ENV CUSPARSELT_PATH="/projects/libcusparse_lt-linux-x86_64-0.3.0.3-archive/"
ENV CUSPARSELT_DIR="/projects/libcusparse_lt-linux-x86_64-0.3.0.3-archive/"
RUN ls $CUSPARSELT_DIR
ENV LD_LIBRARY_PATH=${CUSPARSELT_DIR}/lib64:${LD_LIBRARY_PATH}

# Build venom
RUN git clone --recurse-submodules https://github.com/UDC-GAC/venom.git
#WORKDIR /projects/venom
#RUN mkdir build
#WORKDIR build
#RUN cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="86" && make -j 16

ENV CUDA_INSTALL_PATH=/usr/local/cuda-11.5
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH

# install Python3.9
RUN apt-get install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.9 python3-pip vim

# install python libraries
RUN python3.9 -m pip install numpy matplotlib seaborn pandas shapely

# install anaconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
run rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
ENV PATH=/root/miniconda3/bin:$PATH
RUN python -m pip install numpy matplotlib seaborn pandas shapely

# create end2end venv
RUN conda create -y --name end2end
SHELL ["conda", "run", "-n", "end2end", "/bin/bash", "-c"]
RUN conda install pytorch cudatoolkit torchvision torchaudio pytorch-cuda==11.7 -c pytorch -c nvidia
RUN pip install pybind11
RUN pip install matplotlib shapely holoviews
RUN pip install pandas
RUN pip install seaborn
WORKDIR /projects/venom/end2end/sten
RUN pip install .

# create sparseml venv
WORKDIR /projects/venom/sparseml
RUN conda env create -f sparseml.yml
SHELL ["conda", "run", "-n", "sparseml_artf", "/bin/bash", "-c"]
RUN python3.10 -m pip install -e .
RUN python3.10 -m pip uninstall transformers
RUN python3.10 -m pip install https://github.com/neuralmagic/transformers/releases/download/v1.5/transformers-4.23.1-py3-none-any.whl datasets scikit-learn seqeval pulp

RUN source deactivate

WORKDIR /

#ENV SPUTNIK_PATH=/projects/sputnik
#ENV NCU_PATH=/opt/nvidia/nsight-compute/2020.3.1/ncu

WORKDIR /projects
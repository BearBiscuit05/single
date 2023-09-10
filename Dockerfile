FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# change software source
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
COPY ./docker/pip.conf /root/.pip/pip.conf


# apt software
RUN apt-get update && apt-get install -y wget gnuplot git vim\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# miniconda
WORKDIR /app
COPY . /app/
ENV PATH="/miniconda3/bin:$PATH"
# installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-latest.sh \
    && bash ./miniconda-latest.sh -b -p /miniconda3 \
    && ln -s /miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && find /miniconda3/ -follow -type f -name '*.a' -delete \
    && find /miniconda3/ -follow -type f -name '*.js.map' -delete \
    && conda clean -afy
# create environment

RUN conda create --name sgnn --file /app/docker/spec-list.txt \
    && conda clean -afy \
    && echo "conda activate sgnn" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "fgnn_env", "/bin/bash", "-c"]

# WORKDIR /app/source
# COPY . ./fgnn
# # install dgl
# RUN pip install 'numpy>=1.14.0' 'scipy>=1.1.0' 'networkx>=2.1' 'requests>=2.19.0' \
#     && bash ./fgnn/3rdparty/dgl_install.sh
# # install fastgraph
# RUN bash ./fgnn/utility/fg_install.sh
# # install pyg
# RUN bash ./fgnn/3rdparty/pyg_install.sh

# # install fgnn
# RUN pushd ./fgnn \
#     && bash ./build.sh \
#     && rm -rf build \
#     && rm -rf 3rdparty/dgl/build \
#     && popd \
#     && echo "ulimit -l unlimited" >> ~/.bashrc
FROM ubuntu:18.04

# update
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/ubuntu/@g \
      /etc/apt/sources.list && \
    apt-get clean && apt-get update && apt-get upgrade -y

# install essential
RUN apt-get install -y --no-install-recommends \
      sudo wget curl git build-essential


# create user
ENV USERNAME ubuntu
ENV PASSWORD ubuntu
RUN useradd -m $USERNAME && \
    echo "$USERNAME:$PASSWORD" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod  --uid 1000 $USERNAME && \
    groupmod --gid 1000 $USERNAME


# install dependencies
RUN apt-get install -y --no-install-recommends \
      libyaml-cpp-dev libceres-dev \
      python-dev cython python-scipy python-pillow


# entrypoint
USER $USERNAME
WORKDIR /home/$USERNAME


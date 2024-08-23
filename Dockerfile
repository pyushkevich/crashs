# THE FIRST PART OF THIS Dockerfile is taken from https://github.com/nighres/nighres/blob/master/Dockerfile
# Start from the official Debian image
FROM debian:bullseye

<<<<<<< Updated upstream
# Install nighres prerequisites
RUN echo "deb http://archive.ubuntu.com/ubuntu/ bionic main universe" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
RUN apt-get update
RUN apt-get install -y openjdk-11-jdk python3-jcc

# Install nighres
RUN git clone https://github.com/nighres/nighres /tk/nighres && cd /tk/nighres && git checkout 7293c368476a015708b3a89c43409fdfdef9e19c
RUN ln -s /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java
RUN apt-get install -y wget
WORKDIR /tk/nighres
RUN ./build.sh || ./build.sh
RUN python3 -m pip install .
RUN pip3 install nibabel==3.2.2
=======
# Install necessary tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    openjdk-17-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Adjust JAVA_HOME if necessary and create a symbolic link to match JCC's expected JDK path
# RUN ls -l /usr/lib/jvm && exit 255
ENV JAVA_HOME=/usr/lib/jvm/openjdk-17
RUN ln -s $JAVA_HOME /usr/lib/jvm/temurin-17-jdk-amd64

# Install JCC
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install jcc

# Clone the nighres repository
RUN git clone https://github.com/nighres/nighres

# Change directory into the 	cloned repository, run the build script, and install nighres
WORKDIR /nighres
RUN ./build.sh && \
    python3 -m pip install .

# Install nighres
#RUN git clone https://github.com/nighres/nighres /tk/nighres && cd /tk/nighres && git checkout master
#RUN ln -s /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java
#RUN apt-get install -y wget
#WORKDIR /tk/nighres
#RUN ./build.sh || ./build.sh
#RUN python3 -m pip install .
#RUN pip3 install nibabel==3.2.2
>>>>>>> Stashed changes

# Install python packages
COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy the contents
COPY . /tk/crashs
WORKDIR /tk/crashs

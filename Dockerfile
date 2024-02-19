# Using the 2023a build of our utilities as the base
FROM pyushkevich/tk:2023b

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

# Install python packages
COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy the contents
COPY . /tk/crashs
WORKDIR /tk/crashs

FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

# Install dependencies
RUN apt-get update -y
#RUN apt-get update -y
#RUN pip install --upgrade pip

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libopenjpeg2 \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		python-dev \
		python-tk \
		python-numpy \
		python3-dev \
		python3-tk \
		python3-numpy \
		ant \
		default-jdk \
		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3




RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py


#easy_install pip==1.2.1
#ii Install apache and write hello world message
RUN echo "Hello World!" 
RUN echo $(pip --version)
RUN echo $(python --version)


RUN pip install --upgrade awscli==1.14.5 s3cmd==2.0.1 python-magic sh mysqlclient pandas numpy SQLAlchemy psycopg2-binary
RUN echo $(aws --version)
RUN echo $(aws s3 ls s3://tf-bucket-dev/VIDEOS/)
#RUN aws s3 cp s3://tf-bucket-dev/VIDEOS/test1.mp4 s3://tf-bucket-dev/OUTPUT/

COPY test.py /root/
WORKDIR "/root"

ENTRYPOINT ["python", "test.py"]

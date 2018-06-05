SOURCE_DIR = src
DATA_DIR = data
BUILD_DIR = build
LOGS_DIR = logs
EXEC_FILE = CUDA-DNN-MNIST

CPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp')
GPU_SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cu')

dataset:
	mkdir -p ${DATA_DIR}
	curl -o ${DATA_DIR}/train-images.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
	curl -o ${DATA_DIR}/train-labels.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
	curl -o ${DATA_DIR}/test-images.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	curl -o ${DATA_DIR}/test-labels.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gunzip ${DATA_DIR}/train-images.gz
	gunzip ${DATA_DIR}/train-labels.gz
	gunzip ${DATA_DIR}/test-images.gz
	gunzip ${DATA_DIR}/test-labels.gz

build: FORCE
	mkdir -p ${BUILD_DIR}
	nvcc ${CPU_SOURCE_FILES} ${GPU_SOURCE_FILES} -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	mkdir -p ${LOGS_DIR}
	./${BUILD_DIR}/${EXEC_FILE}

run_experiments:
	mkdir -p ${LOGS_DIR}
	python3 run_experiments.py

clean:
	rm -rf ${BUILD_DIR}

FORCE:

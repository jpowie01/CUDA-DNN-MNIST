SOURCE_DIR = src
DATA_DIR = data
BUILD_DIR = build
EXEC_FILE = CUDA-DNN-MNIST

SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cu')

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
	nvcc ${SOURCE_FILES} -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	./${BUILD_DIR}/${EXEC_FILE}

clean:
	rm -rf ${BUILD_DIR}

FORCE:

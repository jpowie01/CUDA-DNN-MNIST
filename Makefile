SOURCE_DIR=src
BUILD_DIR=build
EXEC_FILE=CUDA-DNN-MNIST

build:
	mkdir -p ${BUILD_DIR}
	nvcc ${SOURCE_DIR}/*.cu -o ${BUILD_DIR}/${EXEC_FILE}

run:
	./${BUILD_DIR}/${EXEC_FILE}

clean:
	rm -rf ${BUILD_DIR}

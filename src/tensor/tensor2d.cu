#include "tensor2d.cuh"

__global__
void kAdd1D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] += b[x];
    }
}

__global__
void kAdd2D(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] += b[y*sizeX + x];
    }
}

__global__
void kSubtract(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] -= b[y*sizeX + x];
    }
}

__global__
void kScale(float *a, float factor, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeX + x] *= factor;
    }
}

__global__
void kMultiply(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
               float* A, int aX, int aY,
               float* B, int bX, int bY,
               float* C)
{
    int outputSizeX = bX;
    int outputSizeY = aY;
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(outputSizeX, blockStartX + fieldsPerBlockX);
    int blockEndY = min(outputSizeY, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < aX; i++) {
                sum += A[y*aX + i] * B[i*bX + x];
            }
            C[y*bX + x] = sum;
        }
    }
}

__global__
void kMultiplyWithSharedMemory(float* A, int aX, int aY,
                               float* B, int bX, int bY,
                               float* C)
{
    int outputSizeX = bX;
    int outputSizeY = aY;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int chunks = (aX + blockDim.x) / blockDim.x;

    if (x >= outputSizeX || y >= outputSizeY) return;

    extern __shared__ float sub[];
    float* As = sub;
    float* Bs = sub + blockDim.x * blockDim.y;

    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; chunk++) {
        // Safely copy data from matrix A
        if (chunk * blockDim.x + threadIdx.x < aX && blockIdx.y * blockDim.y + threadIdx.y < aY) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[(blockIdx.y * blockDim.y + threadIdx.y) * aX + chunk * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Safely copy data from matrix B
        if (blockIdx.x * blockDim.x + threadIdx.x < bX && chunk * blockDim.y + threadIdx.y < bY) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(chunk * blockDim.y + threadIdx.y) * bX + blockIdx.x * blockDim.x + threadIdx.x];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Run calculations on shared memory matrix
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            sum += As[threadIdx.y * blockDim.x + i] * Bs[i * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    C[y*outputSizeX + x] = sum;
}

__global__
void kMultiplyByTransposition(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                              float* A, int aX, int aY,
                              float* B, int bX, int bY,
                              float* C)
{
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(bY, blockStartX + fieldsPerBlockX);
    int blockEndY = min(aY, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < aX; i++) {
                sum += A[y*aX + i] * B[x*bX + i];
            }
            C[y*bY + x] = sum;
        }
    }
}

__global__
void kMultiplyByTranspositionWithSharedMemory(float* A, int aX, int aY,
                                              float* B, int bX, int bY,
                                              float* C)
{
    int outputSizeX = bY;
    int outputSizeY = aY;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int chunks = (aX + blockDim.x) / blockDim.x;

    if (x >= outputSizeX || y >= outputSizeY) return;

    extern __shared__ float sub[];
    float* As = sub;
    float* Bs = sub + blockDim.x * blockDim.y;

    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; chunk++) {
        // Safely copy data from matrix A
        if (chunk * blockDim.x + threadIdx.x < aX && blockIdx.y * blockDim.y + threadIdx.y < aY) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[(blockIdx.y * blockDim.y + threadIdx.y) * aX + chunk * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Safely copy data from matrix B
        if (chunk * blockDim.x + threadIdx.x < bX && blockIdx.x * blockDim.y + threadIdx.y < bY) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(blockIdx.x * blockDim.y + threadIdx.y) * bX + chunk * blockDim.x + threadIdx.x];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
        }

        // Run calculations on shared memory matrix
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            sum += As[threadIdx.y * blockDim.x + i] * Bs[threadIdx.x * blockDim.x + i];
        }
        __syncthreads();
    }
    C[y*outputSizeX + x] = sum;
}

__global__
void kTransposeAndMultiply(int fieldsPerBlockX, int fieldsPerBlockY, int fieldsPerThreadX, int fieldsPerThreadY,
                           float* A, int aX, int aY,
                           float* B, int bX, int bY,
                           float* C)
{
    int blockStartX = blockIdx.x * fieldsPerBlockX;
    int blockStartY = blockIdx.y * fieldsPerBlockY;
    int blockEndX = min(bX, blockStartX + fieldsPerBlockX);
    int blockEndY = min(aX, blockStartY + fieldsPerBlockY);
    int threadStartX = threadIdx.x * fieldsPerThreadX;
    int threadStartY = threadIdx.y * fieldsPerThreadY;
    int threadEndX = threadStartX + fieldsPerThreadX;
    int threadEndY = threadStartY + fieldsPerThreadY;

    int startX = blockStartX + threadStartX;
    int endX = min(blockEndX, blockStartX + threadEndX);
    int startY = blockStartY + threadStartY;
    int endY = min(blockEndY, blockStartY + threadEndY);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            float sum = 0.0f;
            for (int i = 0; i < bY; i++) {
                sum += A[i*aX + y] * B[i*bX + x];
            }
            C[y*bX + x] = sum;
        }
    }
}

__global__
void kTransposeAndMultiplyWithSharedMemory(float* A, int aX, int aY,
                                           float* B, int bX, int bY,
                                           float* C)
{
    int outputSizeX = bX;
    int outputSizeY = aX;
    int elementsInChunk = blockDim.x;  // X & Y should be equal!
    int x = blockIdx.x * elementsInChunk + threadIdx.x;
    int y = blockIdx.y * elementsInChunk + threadIdx.y;
    int chunks = (aY + elementsInChunk) / elementsInChunk;

    if (x >= outputSizeX || y >= outputSizeY) return;

    extern __shared__ float sub[];
    float* As = sub;
    float* Bs = sub + elementsInChunk * elementsInChunk;

    float sum = 0.0f;
    for (int chunk = 0; chunk < chunks; chunk++) {
        // Safely copy data from matrix A
        if (blockIdx.y * elementsInChunk + threadIdx.x < aX && chunk * elementsInChunk + threadIdx.y < aY) {
            As[threadIdx.y * elementsInChunk + threadIdx.x] = 
                A[(chunk * elementsInChunk + threadIdx.y) * aX + blockIdx.y * elementsInChunk + threadIdx.x];
        } else {
            As[threadIdx.y * elementsInChunk + threadIdx.x] = 0.0;
        }

        // Safely copy data from matrix B
        if (blockIdx.x * elementsInChunk + threadIdx.x < bX && chunk * elementsInChunk + threadIdx.y < bY) {
            Bs[threadIdx.y * elementsInChunk + threadIdx.x] =
                B[(chunk * elementsInChunk + threadIdx.y) * bX + blockIdx.x * elementsInChunk + threadIdx.x];
        } else {
            Bs[threadIdx.y * elementsInChunk + threadIdx.x] = 0.0;
        }

        // Run calculations on shared memory matrix
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            sum += As[i * elementsInChunk + threadIdx.y] * Bs[i * elementsInChunk + threadIdx.x];
        }
        __syncthreads();
    }
    C[y*outputSizeX + x] = sum;
}

__global__
void kMeanX(float* a, int aX, int aY, float* b)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < aX) {
        float sum = 0.0;
        for (int i = 0; i < aY; i++) {
            sum += a[i*aX + col];
        }
        b[col] = sum / aY;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    if (this->sizeX && this->sizeY) {
        cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
    } else {
        this->devData = NULL;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float** hostData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    if (this->sizeX && this->sizeY) {
        cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
        cudaMemcpy(this->devData, *hostData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        this->devData = NULL;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float* devData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->devData = devData;
}

Tensor2D::~Tensor2D() {
    cudaFree(this->devData);
}

int Tensor2D::getSize(Tensor2DAxis axis) {
    if (axis == X) {
        return this->sizeX;
    } else if (axis == Y) {
        return this->sizeY;
    }
    return -1;
}

float* Tensor2D::getDeviceData() {
    return this->devData;
}

float** Tensor2D::fetchDataFromDevice() {
    float** hostData = new float*[this->sizeY];
    *hostData = new float[this->sizeY * this->sizeX];
    for (int i = 1; i < this->sizeY; i++) hostData[i] = hostData[i-1] + this->sizeX;
    cudaMemcpy(*hostData, this->devData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor2D::add(Tensor1D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize()) {
        printf("ERROR! Cannot add vector with size %d to matrix %dx%d.\n",
               tensor->getSize(), this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensor2DAddBlockSize, Configuration::tensor2DAddBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd1D<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::add(Tensor2D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(X) || this->sizeY != tensor->getSize(Y)) {
        printf("ERROR! Cannot add matrix with size %dx%d to matrix %dx%d.\n",
               tensor->getSize(X), tensor->getSize(Y), this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensor2DAddBlockSize, Configuration::tensor2DAddBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd2D<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::subtract(Tensor2D* tensor) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(X) || this->sizeY != tensor->getSize(Y)) {
        printf("ERROR! Cannot subtract matrix with size %dx%d to matrix %dx%d.\n",
               tensor->getSize(X), tensor->getSize(Y), this->sizeX, this->sizeY);
        exit(1);
    }

    // Defer calculations on GPU
    dim3 threadsPerBlock(Configuration::tensor2DSubtractBlockSize, Configuration::tensor2DSubtractBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kSubtract<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::scale(float factor) {
    dim3 threadsPerBlock(Configuration::tensor2DScaleBlockSize, Configuration::tensor2DScaleBlockSize);
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kScale<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), factor, this->sizeX, this->sizeY);
}

Tensor2D* Tensor2D::multiply(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(Y)) {
        printf("ERROR! Cannot multiply matrices with shape %dx%d and %dx%d.\n",
               this->sizeX, this->sizeY, tensor->getSize(X), tensor->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensor2DMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensor2DMultiplyBlockSize, Configuration::tensor2DMultiplyBlockSize);
        dim3 numBlocks((tensor->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kMultiplyWithSharedMemory<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensor2DMultiplyBlockSize;
        int threadsY = Configuration::tensor2DMultiplyBlockSize;
        int blocksX = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (tensor->getSize(X) + threadsX) / threadsX
                       : Configuration::tensor2DMultiplyBlockNumber;
        int blocksY = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (this->sizeY + threadsY) / threadsY
                       : Configuration::tensor2DMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor->getSize(Y) + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->getSize(Y) + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
        kMultiply<<<numBlocks, threadsPerBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor2D* Tensor2D::multiplyByTransposition(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeX != tensor->getSize(X)) {
        printf("ERROR! Cannot multiply matrix with shape %dx%d by transposition of matrix %dx%d.\n",
               this->sizeX, this->sizeY, tensor->getSize(X), tensor->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensor2DMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensor2DMultiplyBlockSize, Configuration::tensor2DMultiplyBlockSize);
        dim3 numBlocks((tensor->getSize(Y) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kMultiplyByTranspositionWithSharedMemory<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensor2DMultiplyBlockSize;
        int threadsY = Configuration::tensor2DMultiplyBlockSize;
        int blocksX = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (tensor->getSize(Y) + threadsX) / threadsX
                       : Configuration::tensor2DMultiplyBlockNumber;
        int blocksY = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (this->sizeY + threadsY) / threadsY
                       : Configuration::tensor2DMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor->getSize(Y) + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->getSize(Y) + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
        kMultiplyByTransposition<<<numBlocks, threadsPerBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor2D* Tensor2D::transposeAndMultiply(Tensor2D* tensor, Tensor2D* output) {
    // Check sizes and exit program in case of invalid multiplication
    if (this->sizeY != tensor->getSize(Y)) {
        printf("ERROR! Cannot multiply transposition of matrix with shape %dx%d by matrix %dx%d.\n",
               this->sizeX, this->sizeY, tensor->getSize(X), tensor->getSize(Y));
        exit(1);
    }

    // In case of using shared memory, we've got to use dynamic amount of blocks
    if (Configuration::tensor2DMultiplySharedMemory == 1) {
        // Prepare configuration for CUDA kernel
        dim3 threadsPerBlock(Configuration::tensor2DMultiplyBlockSize, Configuration::tensor2DMultiplyBlockSize);
        dim3 numBlocks((tensor->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                       (this->sizeX + threadsPerBlock.y)/threadsPerBlock.y);
        int sharedMemorySize = 2 * threadsPerBlock.y * threadsPerBlock.x * sizeof(float);

        // Defer calculations on GPU
        kTransposeAndMultiplyWithSharedMemory<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    } else {
        // Prepare configuration for CUDA kernel
        int threadsX = Configuration::tensor2DMultiplyBlockSize;
        int threadsY = Configuration::tensor2DMultiplyBlockSize;
        int blocksX = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (tensor->getSize(X) + threadsX) / threadsX
                       : Configuration::tensor2DMultiplyBlockNumber;
        int blocksY = Configuration::tensor2DMultiplyBlockNumber == -1
                       ? (this->getSize(X) + threadsY) / threadsY
                       : Configuration::tensor2DMultiplyBlockNumber;
        int fieldsPerBlockX = max(1, (tensor->getSize(X) + blocksX) / blocksX);
        int fieldsPerThreadX = max(1, (fieldsPerBlockX + threadsX) / threadsX);
        int fieldsPerBlockY = max(1, (this->getSize(X) + blocksY) / blocksY);
        int fieldsPerThreadY = max(1, (fieldsPerBlockY + threadsY) / threadsY);
        dim3 threadsPerBlock(threadsX, threadsY);
        dim3 numBlocks(blocksX, blocksY);

        // Defer calculations on GPU
        kTransposeAndMultiply<<<numBlocks, threadsPerBlock>>>(
            fieldsPerBlockX, fieldsPerBlockY, fieldsPerThreadX, fieldsPerThreadY,
            this->getDeviceData(), this->sizeX, this->sizeY,
            tensor->getDeviceData(), tensor->getSize(X), tensor->getSize(Y),
            output->getDeviceData()
        );
    }
    return output;
}

Tensor1D* Tensor2D::meanX(Tensor1D* output) {
    int threadsPerBlock = Configuration::tensor2DMeanBlockSize;
    int numBlocks = (this->sizeX + threadsPerBlock)/threadsPerBlock;
    kMeanX<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, output->getDeviceData());
    return output;
}

void Tensor2D::debugPrint() {
    float** values = this->fetchDataFromDevice();
    for (int y = 0; y < this->sizeY; y++) {
        for (int x = 0; x < this->sizeX; x++) {
            printf("%8.5f; ", values[y][x]);
        }
        printf("\n");
    }
    delete[] values;
}

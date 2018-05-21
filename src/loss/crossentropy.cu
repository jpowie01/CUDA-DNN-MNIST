#include "crossentropy.h"

__global__
void kSoftMaxCrossEntropy(float *output, int oX, int oY, float* labels, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < 0.0000001) {
            sum = 0.0000001;
        }

        for (int i = 0; i < oX; i++) {
            // Softmax = exp(value) / sum(exp(allValues))
            // Subtract truth (which is one hot)
            y[row*oX + i] = (exp(output[row*oX + i]) / sum) - labels[row*oX + i];
        }
    }
}

CrossEntropyLoss::CrossEntropyLoss() {
    // TODO: ...
}

float CrossEntropyLoss::getLoss(Tensor2D* networkOutput, Tensor2D* labels) {
    float** output = networkOutput->fetchDataFromDevice();  // TODO: Potential memory leak...
    float** target = labels->fetchDataFromDevice();  // TODO: Potential memory leak...

    float error = 0.0;
    float sum = 0.0;
    for (int x = 0; x < networkOutput->sizeX; x++) {
        sum += exp(output[0][x]);
    }
    if (abs(sum) < 0.0000001) {
        sum = 0.0000001;
    }
    for (int x = 0; x < networkOutput->sizeX; x++) {
        error -= target[0][x] * log(exp(output[0][x]) / sum);  // TODO: Iterate over batch
    }
    return error;
}

Tensor2D* CrossEntropyLoss::calculate(Tensor2D* networkOutput, Tensor2D* labels) {
    float* output;
    cudaMalloc((void **)&(output), networkOutput->sizeX*networkOutput->sizeY*sizeof(float));

    dim3 threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((networkOutput->sizeY + threadsPerBlock.x)/threadsPerBlock.x);
    kSoftMaxCrossEntropy<<<numBlocks, threadsPerBlock>>>(networkOutput->getDeviceData(), networkOutput->sizeX, networkOutput->sizeY, labels->getDeviceData(), output);

    return new Tensor2D(networkOutput->sizeX, networkOutput->sizeY, output);
}

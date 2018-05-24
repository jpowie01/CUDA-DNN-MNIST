#include "crossentropy.h"

#define VERY_SMALL_NUMBER 1e-10

__global__
void kSoftMaxCrossEntropy(float *output, int oX, int oY, float* labels, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < VERY_SMALL_NUMBER) {
            sum = VERY_SMALL_NUMBER;
        }

        for (int i = 0; i < oX; i++) {
            // Softmax = exp(value) / sum(exp(allValues))
            // Subtract truth (which is one hot)
            y[row*oX + i] = (exp(output[row*oX + i]) / sum) - labels[row*oX + i];
        }
    }
}

__global__
void kSoftMaxCrossEntropyLoss(float *output, int oX, int oY, float* labels, float* error) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < oX; i++) {
            sum += exp(output[row*oX + i]);
        }
        if (abs(sum) < VERY_SMALL_NUMBER) {
            sum = VERY_SMALL_NUMBER;
        }

        float tmpError = 0.0;
        for (int i = 0; i < oX; i++) {
            // Error = target * log(softmaxOutput) + (1 - target) * log (1 - softmaxOutput)
            tmpError -= labels[row*oX + i] * log(exp(output[row*oX + i]) / sum) + 
                        (1 - labels[row*oX + i]) * log(1 - exp(output[row*oX + i]) / sum);
        }
        atomicAdd(error, tmpError);
    }
}

__global__
void kSoftMaxCrossEntropyAccuracy(float *output, int oX, int oY, float* labels, float* accuracy) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < oY) {
        int maxIdx = 0;
        float maxValue = output[row*oX];
        for (int x = 1; x < oX; x++) {
            if (output[row*oX + x] > maxValue) {
                maxIdx = x;
                maxValue = output[row*oX + x];
            }
        }
        if (output[row*oX + maxIdx] > 1.0 - VERY_SMALL_NUMBER) {
            atomicAdd(accuracy, 1);
        }
    }
}

CrossEntropyLoss::CrossEntropyLoss() {}

float CrossEntropyLoss::getLoss(Tensor2D* networkOutput, Tensor2D* labels) {
    float error = 0.0;
    float* dError;
    cudaMalloc((void**)&dError, sizeof(float));
    cudaMemcpy(dError, &error, sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((networkOutput->sizeY + threadsPerBlock.x)/threadsPerBlock.x);
    kSoftMaxCrossEntropyLoss<<<numBlocks, threadsPerBlock>>>(networkOutput->getDeviceData(), networkOutput->sizeX, networkOutput->sizeY, labels->getDeviceData(), dError);
    cudaMemcpy(&error, dError, sizeof(float), cudaMemcpyDeviceToHost);
    return error / networkOutput->sizeY;
}

float CrossEntropyLoss::getAccuracy(Tensor2D* networkOutput, Tensor2D* labels) {
    float accuracy = 0.0;
    float* dAccuracy;
    cudaMalloc((void**)&dAccuracy, sizeof(float));
    cudaMemcpy(dAccuracy, &accuracy, sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((networkOutput->sizeY + threadsPerBlock.x)/threadsPerBlock.x);
    kSoftMaxCrossEntropyAccuracy<<<numBlocks, threadsPerBlock>>>(networkOutput->getDeviceData(), networkOutput->sizeX, networkOutput->sizeY, labels->getDeviceData(), dAccuracy);
    cudaMemcpy(&accuracy, dAccuracy, sizeof(float), cudaMemcpyDeviceToHost);
    return 100.0 * accuracy / networkOutput->sizeY;
}

Tensor2D* CrossEntropyLoss::calculate(Tensor2D* networkOutput, Tensor2D* labels) {
    float* output;
    cudaMalloc((void **)&(output), networkOutput->sizeX*networkOutput->sizeY*sizeof(float));

    dim3 threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((networkOutput->sizeY + threadsPerBlock.x)/threadsPerBlock.x);
    kSoftMaxCrossEntropy<<<numBlocks, threadsPerBlock>>>(networkOutput->getDeviceData(), networkOutput->sizeX, networkOutput->sizeY, labels->getDeviceData(), output);

    return new Tensor2D(networkOutput->sizeX, networkOutput->sizeY, output);
}

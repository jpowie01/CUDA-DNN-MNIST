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

CrossEntropyLoss::CrossEntropyLoss() {}

float CrossEntropyLoss::getLoss(Tensor2D* networkOutput, Tensor2D* labels) {
    float** output = networkOutput->fetchDataFromDevice();
    float** target = labels->fetchDataFromDevice();
    float totalSumOfErrors = 0.0;

    for (int y = 0; y < networkOutput->sizeY; y++) {
        float error = 0.0;
        float sum = 0.0;
        for (int x = 0; x < networkOutput->sizeX; x++) {
            sum += exp(output[y][x]);
        }
        if (abs(sum) < 1e-10) {
            sum = 1e-10;
        }
        for (int x = 0; x < networkOutput->sizeX; x++) {
            error -= target[y][x] * log(exp(output[y][x]) / sum) + (1 - target[y][x]) * log(exp(output[y][x]) / sum);
        }
        totalSumOfErrors += error;
    }

    // Clean memory and return output
    delete[] output;
    delete[] target;
    return totalSumOfErrors / networkOutput->sizeY;
}

float CrossEntropyLoss::getAccuracy(Tensor2D* networkOutput, Tensor2D* labels) {
    float** output = networkOutput->fetchDataFromDevice();
    float** target = labels->fetchDataFromDevice();
    float totalMatched = 0.0;

    for (int y = 0; y < networkOutput->sizeY; y++) {
        int maxIdx = 0;
        float maxValue = output[y][0];
        for (int x = 1; x < networkOutput->sizeX; x++) {
            if (output[y][x] > maxValue) {
                maxIdx = x;
                maxValue = output[y][x];
            }
        }
        if (target[y][maxIdx] > 1.0 - 1.0e-10) {
            totalMatched += 1;
        }
    }

    // Clean memory and return output
    delete[] output;
    delete[] target;
    return 100.0 * totalMatched / networkOutput->sizeY;
}

Tensor2D* CrossEntropyLoss::calculate(Tensor2D* networkOutput, Tensor2D* labels) {
    float* output;
    cudaMalloc((void **)&(output), networkOutput->sizeX*networkOutput->sizeY*sizeof(float));

    dim3 threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((networkOutput->sizeY + threadsPerBlock.x)/threadsPerBlock.x);
    kSoftMaxCrossEntropy<<<numBlocks, threadsPerBlock>>>(networkOutput->getDeviceData(), networkOutput->sizeX, networkOutput->sizeY, labels->getDeviceData(), output);

    return new Tensor2D(networkOutput->sizeX, networkOutput->sizeY, output);
}

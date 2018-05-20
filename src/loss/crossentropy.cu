#include "crossentropy.h"

__global__
void kSoftMaxCrossEntropy(float *a, int aX, int aY, float* labels, float* b) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < aY) {
        /*
        // Calculate sum of exponents for whole column
        float sum = 0.0;
        for (int i = 0; i < aX; i++) {
            sum += exp(a[row*aX + i]);
        }
        if (abs(sum) < 0.0000001) {
            sum = 0.0000001;
        }
        */

        for (int i = 0; i < aX; i++) {
            /*
            // Softmax = exp(value) / sum(exp(allValues))
            b[row*aX + i] = exp(a[row*aX + i]) / sum;
            */

            // Subtract truth (which is one hot)
            b[row*aX + i] = a[row*aX + i] - labels[row*aX + i];

            // Normalize
            //b[row*aX + i] /= aX;
        }
    }
}

CrossEntropyLoss::CrossEntropyLoss() {
    // TODO: ...
}

Tensor2D* CrossEntropyLoss::calculate(Tensor2D* networkOutput, Tensor2D* labels) {
    float* output;
    cudaMalloc((void **)&(output), networkOutput->sizeX*networkOutput->sizeY*sizeof(float));

    dim3 threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((networkOutput->sizeY + threadsPerBlock.x)/threadsPerBlock.x);
    kSoftMaxCrossEntropy<<<numBlocks, threadsPerBlock>>>(networkOutput->getDeviceData(), networkOutput->sizeX, networkOutput->sizeY, labels->getDeviceData(), output);

    return new Tensor2D(networkOutput->sizeX, networkOutput->sizeY, output);
}

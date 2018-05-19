#include <stdio.h>
#include <cstdlib>
#include <ctime>

#include "utils.h"
#include "dense.h"


int main() {
    // Always initialize seed to some random value
    // TODO: Maybe it is worth to fix it for experiments?
    srand(static_cast<unsigned>(time(0)));

    // Prepare some example input data - for now it is just random noise
    float** rawExampleData = new float*[16];
    *rawExampleData = new float[16*28*28];
    for (int i = 1; i < 16; i++) rawExampleData[i] = rawExampleData[i-1] + 28*28;
    for (int batch = 0; batch < 16; batch++) {
        for (int i = 0; i < 28*28; i++) {
            rawExampleData[batch][i] = randomFloat(-1.0, 1.0);
        }
    }
    Tensor2D* exampleData = new Tensor2D(28*28, 16, rawExampleData);

    // Prepare all layers
    DenseLayer* inputLayer = new DenseLayer(28*28, 1024);
    DenseLayer* hiddenLayer = new DenseLayer(1024, 1024);
    DenseLayer* outputLayer = new DenseLayer(1024, 10);

    // Forward pass
    printf("\nForward => First Layer:\n");
    Tensor2D* afterInput = inputLayer->forward(exampleData);
    printf("\nForward => Second Layer:\n");
    Tensor2D* afterHidden = hiddenLayer->forward(afterInput);
    printf("\nForward => Third Layer:\n");
    Tensor2D* afterOutput = outputLayer->forward(afterHidden);

    // Output for this example
    printf("\nClassification:\n");
    float** classification = afterOutput->fetchDataFromDevice();
    for (int y = 0; y < 16; y++) {
        printf("Image %d => ", y);
        for (int x = 0; x < 10; x++) {
            printf("[%d]: %.2f; ", x, classification[y][x]);
        }
        printf("\n");
    }

    // Backward pass
    printf("\nBackward => Third Layer:\n");
    Tensor2D* backwardFromOutput = outputLayer->backward(afterOutput);
    printf("\nBackward => Second Layer:\n");
    Tensor2D* backwardFromHidden = hiddenLayer->backward(backwardFromOutput);
    printf("\nBackward => First Layer:\n");
    Tensor2D* backwardFromInput = inputLayer->backward(backwardFromHidden);

    // Clean memory and exit
    /*
    delete tensorA;
    delete tensorB;
    delete data;
    delete b;
    delete a;
    */
    return 0;
}

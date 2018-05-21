#include <stdio.h>
#include <cstdlib>
#include <ctime>

#include "layers/dense.h"
#include "layers/relu.h"
#include "optimizers/sgd.h"
#include "loss/crossentropy.h"
#include "models/sequential.h"
#include "utils.h"


int main() {
    // Always initialize seed to some random value
    // TODO: Maybe it is worth to fix it for experiments?
    //srand(static_cast<unsigned>(time(0)));
    srand(123123123);

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

    // Prepare some example labels for above input data - for now it is just random noise
    float** rawExampleLabels = new float*[16];
    *rawExampleLabels = new float[16*10];
    for (int i = 1; i < 16; i++) rawExampleLabels[i] = rawExampleLabels[i-1] + 10;
    for (int batch = 0; batch < 16; batch++) {
        int randomLabel = randomInt(0, 9);
        for (int i = 0; i < 10; i++) {
            rawExampleLabels[batch][i] = 0;
        }
        rawExampleLabels[batch][randomLabel] = 1;
    }
    Tensor2D* labels = new Tensor2D(10, 16, rawExampleLabels);

    // Prepare optimizer and loss function
    SGDOptimizer* optimizer = new SGDOptimizer(0.001);
    CrossEntropyLoss* loss = new CrossEntropyLoss();

    // Prepare model
    SequentialModel* model = new SequentialModel(optimizer, loss);
    model->addLayer(new DenseLayer(28*28, 300));
    model->addLayer(new ReLuLayer(300));
    model->addLayer(new DenseLayer(300, 300));
    model->addLayer(new ReLuLayer(300));
    model->addLayer(new DenseLayer(300, 10));

    // Run some epochs
    int epochs = 3000;  // TODO: Put it somewhere else to simplify experiments!
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        Tensor2D* output = model->forward(exampleData);

        // Output for this example
        // printf("\nClassification:\n");
        // output->debugPrint();

        // Output for this example
        // printf("\nLabels:\n");
        // labels->debugPrint();

        // Print error
        printf("Epoch: %d\tError: %.5f\n", epoch, loss->getLoss(output, labels));

        // Backward pass
        model->backward(output, labels);
    }

    // TODO: Clean memory and exit
    /*
    delete tensorA;
    delete tensorB;
    delete data;
    delete b;
    delete a;
    */
    return 0;
}

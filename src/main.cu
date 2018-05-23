#include <stdio.h>
#include <cstdlib>
#include <ctime>

#include "layers/dense.h"
#include "layers/relu.h"
#include "optimizers/sgd.h"
#include "loss/crossentropy.h"
#include "models/sequential.h"
#include "datasets/mnist.h"
#include "utils.h"


int main() {
    // Always initialize seed to some random value
    //srand(static_cast<unsigned>(time(0)));
    srand(123123123);

    MNISTDataSet* dataset = new MNISTDataSet();

    // Prepare optimizer and loss function
    SGDOptimizer* optimizer = new SGDOptimizer(0.000000001);
    CrossEntropyLoss* loss = new CrossEntropyLoss();

    // Prepare model
    SequentialModel* model = new SequentialModel(optimizer, loss);
    model->addLayer(new DenseLayer(28*28, 500));
    model->addLayer(new ReLuLayer(500));
    model->addLayer(new DenseLayer(500, 300));
    model->addLayer(new ReLuLayer(300));
    model->addLayer(new DenseLayer(300, 10));

    // Run some epochs
    int epochs = 30;  // TODO: Put it somewhere else to simplify experiments!
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int batch = 0; batch < dataset->getSize() / 16; batch++) {
            // Fetch batch from dataset
            Tensor2D* images = dataset->getBatchOfImages(batch, 16);
            Tensor2D* labels = dataset->getBatchOfLabels(batch, 16);

            // Forward pass
            Tensor2D* output = model->forward(images);

            // Print error
            printf("Epoch: %d\tBatch: %d\tError: %8.5f\tAccuracy: %8.5f%%\n", epoch, batch, loss->getLoss(output, labels), loss->getAccuracy(output, labels));

            // Backward pass
            model->backward(output, labels);

            // Clean data for this batch
            delete images;
            delete labels;
        }
    }
    return 0;
}

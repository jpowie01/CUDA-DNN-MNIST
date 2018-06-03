#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "layers/dense.hpp"
#include "layers/relu.cuh"
#include "optimizers/sgd.hpp"
#include "loss/crossentropy.cuh"
#include "models/sequential.cuh"
#include "datasets/mnist.hpp"
#include "utils.hpp"
#include "configuration.hpp"


int main() {
    // Always initialize seed to some random value
    clock_t start, end;
    srand(static_cast<unsigned>(time(0)));

    // Print our current configuration for this training
    Configuration::printCurrentConfiguration();
    Configuration::printCUDAConfiguration();

    // Read both training and test dataset
    MNISTDataSet* trainDataset = new MNISTDataSet(TRAIN);
    MNISTDataSet* testDataset = new MNISTDataSet(TEST);

    // Prepare optimizer and loss function
    float learningRate = Configuration::learningRate;
    SGDOptimizer* optimizer = new SGDOptimizer(learningRate);
    CrossEntropyLoss* loss = new CrossEntropyLoss();

    // Prepare model
    SequentialModel* model = new SequentialModel(optimizer, loss);
    model->addLayer(new DenseLayer(28*28, 100));
    model->addLayer(new ReLuLayer(100));
    model->addLayer(new DenseLayer(100, 10));

    // Run some epochs
    int epochs = Configuration::numberOfEpochs;
    int batchSize = Configuration::batchSize;
    int numberOfTrainBatches = trainDataset->getSize() / batchSize;
    int numberOfTestBatches = testDataset->getSize() / batchSize;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float trainingLoss = 0.0, trainingAccuracy = 0.0;
        double trainingForwardTime = 0.0, trainingBackwardTime = 0.0;
        printf("Epoch %d:\n", epoch);
        for (int batch = 0; batch < numberOfTrainBatches; batch++) {
            // Fetch batch from dataset
            Tensor2D* images = trainDataset->getBatchOfImages(batch, batchSize);
            Tensor2D* labels = trainDataset->getBatchOfLabels(batch, batchSize);

            // Forward pass
            start = clock();
            Tensor2D* output = model->forward(images, SYNCHRONIZE_FORWARD);
            end = clock();

            // Save statistics
            trainingLoss += loss->getLoss(output, labels);
            trainingAccuracy += loss->getAccuracy(output, labels);
            trainingForwardTime += double(end - start) / CLOCKS_PER_SEC;

            // Backward pass
            start = clock();
            model->backward(output, labels, SYNCHRONIZE_BACKWARD);
            end = clock();

            // Save statistics
            trainingBackwardTime += double(end - start) / CLOCKS_PER_SEC;

            // Clean data for this batch
            delete images;
            delete labels;
        }

        // Calculate mean training metrics
        trainingLoss /= numberOfTrainBatches;
        trainingAccuracy /= numberOfTrainBatches;
        printf("  - [Train] Loss=%.5f\n", trainingLoss);
        printf("  - [Train] Accuracy=%.5f%%\n", trainingAccuracy);
        printf("  - [Train] Total Forward Time=%.5fms\n", trainingForwardTime * 1000.0);
        printf("  - [Train] Total Backward Time=%.5fms\n", trainingBackwardTime * 1000.0);
        printf("  - [Train] Batch Forward Time=%.5fms\n", trainingForwardTime * 1000.0 / numberOfTrainBatches);
        printf("  - [Train] Batch Backward Time=%.5fms\n", trainingBackwardTime * 1000.0 / numberOfTrainBatches);

        // Check model performance on test set
        float testLoss = 0.0, testAccuracy = 0.0;
        for (int batch = 0; batch < numberOfTestBatches; batch++) {
            // Fetch batch from dataset
            Tensor2D* images = testDataset->getBatchOfImages(batch, batchSize);
            Tensor2D* labels = testDataset->getBatchOfLabels(batch, batchSize);

            // Forward pass
            Tensor2D* output = model->forward(images);

            // Print error
            testLoss += loss->getLoss(output, labels);
            testAccuracy += loss->getAccuracy(output, labels);

            // Clean data for this batch
            delete images;
            delete labels;
        }

        // Calculate mean testing metrics
        testLoss /= numberOfTestBatches;
        testAccuracy /= numberOfTestBatches;
        printf("  - [Test] Loss=%.5f\n", testLoss);
        printf("  - [Test] Accuracy=%.5f%%\n", testAccuracy);
        printf("\n");

        // Shuffle both datasets before next epoch!
        trainDataset->shuffle();
        testDataset->shuffle();
    }
    return 0;
}

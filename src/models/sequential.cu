#include "sequential.h"

SequentialModel::SequentialModel(Optimizer* optimizer, LossFunction* lossFunction) {
    this->optimizer = optimizer;
    this->lossFunction = lossFunction;
}

void SequentialModel::addLayer(Layer* layer) {
    printf("Adding Layer to the model: %d\n", layer);
    this->layers.push_back(layer);
}

Tensor2D* SequentialModel::forward(Tensor2D* input) {
    Tensor2D* values = input;
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
        values = (*layer)->forward(values);  // TODO: Possible memory leak!
        
        // Output for this example
        printf("\nForward pass for Layer %d:\n", (*layer));
        float** example = values->fetchDataFromDevice();
        for (int y = 0; y < 16; y++) {
            printf("Image %d => ", y);
            for (int x = 0; x < 10; x++) {
                printf("[%d]: %.5f; ", x, example[y][x]);
            }
            printf("\n");
        }

    }
    return values;
}

void SequentialModel::backward(Tensor2D* output, Tensor2D* labels) {
    // Compute gradients with loss function
    Tensor2D* gradients = this->lossFunction->calculate(output, labels);

        // Output for this example
        printf("\nBackward pass gradients:\n");
        float** example = gradients->fetchDataFromDevice();
        for (int y = 0; y < 16; y++) {
            printf("Image %d => ", y);
            for (int x = 0; x < 10; x++) {
                printf("[%d]: %.5f; ", x, example[y][x]);
            }
            printf("\n");
        }

    // Pass these gradients with backpropagation
    Tensor2D* values = gradients;
    for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); ++layer) {
        values = (*layer)->backward(values);  // TODO: Possible memory leak!

        // Output for this example
        printf("\nBackward pass for Layer %d:\n", (*layer));
        float** example = values->fetchDataFromDevice();
        for (int y = 0; y < 16; y++) {
            printf("Image %d => ", y);
            for (int x = 0; x < 10; x++) {
                printf("[%d]: %.5f; ", x, example[y][x]);
            }
            printf("\n");
        }
    }

    // Updates all layers with optimizer
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
        optimizer->optimize(*layer);
    }
}

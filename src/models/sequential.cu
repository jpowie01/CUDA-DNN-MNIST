#include "sequential.cuh"

SequentialModel::SequentialModel(Optimizer* optimizer, LossFunction* lossFunction) {
    this->optimizer = optimizer;
    this->lossFunction = lossFunction;
    this->gradients = NULL;
}

void SequentialModel::addLayer(Layer* layer) {
    DEBUG_PRINT("Adding Layer to the model: %d\n", layer);
    this->layers.push_back(layer);
}

Tensor2D* SequentialModel::forward(Tensor2D* input) {
    Tensor2D* values = input;
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        values = (*layer)->forward(values);
        #if defined(DEBUG) && DEBUG >= 2
        DEBUG_PRINT("Forward pass for Layer %d:\n", (*layer));
        values->debugPrint();
        #endif
    }
    return values;
}

void SequentialModel::backward(Tensor2D* output, Tensor2D* labels) {
    // Compute gradients with loss function
    if (!this->gradients) {
        this->gradients = new Tensor2D(output->getSize(X), output->getSize(Y));
    }
    this->lossFunction->calculate(output, labels, this->gradients);
    #if defined(DEBUG) && DEBUG >= 2
    DEBUG_PRINT("Backward pass gradients:\n");
    gradients->debugPrint();
    #endif

    // Pass these gradients with backpropagation
    Tensor2D* values = gradients;
    for (std::vector<Layer*>::reverse_iterator layer = layers.rbegin(); layer != layers.rend(); layer++) {
        values = (*layer)->backward(values);
        #if defined(DEBUG) && DEBUG >= 2
        DEBUG_PRINT("\nBackward pass for Layer %d:\n", (*layer));
        values->debugPrint();
        #endif
    }

    // Updates all layers with optimizer
    for (std::vector<Layer*>::iterator layer = layers.begin(); layer != layers.end(); layer++) {
        optimizer->optimize(*layer);
    }
}

#pragma once
#ifndef SEQUENTIAL_MODEL_HPP
#define SEQUENTIAL_MODEL_HPP

#include <cstdio>
#include <cmath>
#include <vector>

#include "../layers/layer.hpp"
#include "../optimizers/optimizer.hpp"
#include "../loss/lossfunction.hpp"
#include "../tensor/tensor2d.cuh"
#include "../utils.hpp"

#define SYNCHRONIZE_FORWARD     true
#define SYNCHRONIZE_BACKWARD    true


class SequentialModel {
private:
    Optimizer* optimizer;
    LossFunction* lossFunction;
    std::vector<Layer*> layers;

    Tensor2D* gradients;

public:
    SequentialModel(Optimizer* optimizer, LossFunction* lossFunction);

    void addLayer(Layer* layer);
    Tensor2D* forward(Tensor2D* input, bool synchronize = false);
    void backward(Tensor2D* output, Tensor2D* layers, bool synchronize = false);
};

#endif  /* !SEQUENTIAL_MODEL_HPP */

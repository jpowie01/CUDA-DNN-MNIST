#pragma once
#ifndef LAYER_HPP
#define LAYER_HPP

#include <cstdio>
#include <cmath>

#include "../tensor/tensor1d.cuh"
#include "../tensor/tensor2d.cuh"

class Layer {
protected:
    Tensor2D* weights;
    Tensor1D* bias;
    Tensor2D* deltaWeights;
    Tensor1D* deltaBias;

public:
    Tensor2D* getWeights();
    Tensor1D* getBias();
    Tensor2D* getDeltaWeights();
    Tensor1D* getDeltaBias();

    virtual Tensor2D* forward(Tensor2D* data) = 0;
    virtual Tensor2D* backward(Tensor2D* gradients) = 0;
};

#endif  /* !LAYER_HPP */

#pragma once
#ifndef SGD_HPP
#define SGS_HPP

#include <cstdio>
#include <cmath>

#include "../layers/layer.hpp"
#include "../tensor/tensor2d.cuh"
#include "optimizer.hpp"


class SGDOptimizer: public Optimizer {
private:
    float learningRate;

public:
    SGDOptimizer(float learningRate);

    void optimize(Layer* layer);
};

#endif  /* !SGD_HPP */

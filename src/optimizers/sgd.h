#pragma once
#ifndef SGD_H
#define SGS_H

#include <stdio.h>
#include <cmath>

#include "../layers/layer.h"
#include "../tensor/tensor2d.h"
#include "optimizer.h"


class SGDOptimizer: public Optimizer {
private:
    float learningRate;

public:
    SGDOptimizer(float learningRate);

    void optimize(Layer* layer);
};

#endif  /* !SGD_H */

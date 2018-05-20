#pragma once
#ifndef SGD_H
#define SGS_H

#include <stdio.h>
#include <cmath>

#include "../layers/base.h"
#include "../tensor/tensor2d.h"
#include "base.h"


class SGDOptimizer: public Optimizer {
private:
    float learningRate;

public:
    SGDOptimizer(float learningRate);

    void optimize(Layer* layer);
};

#endif  /* !SGD_H */

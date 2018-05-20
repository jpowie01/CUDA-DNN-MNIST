#pragma once
#ifndef BASE_OPTIMIZER_H
#define BASE_OPTIMIZER_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor2d.h"

class Optimizer {
public:
    virtual void optimize(Layer* layer) = 0;
};

#endif  /* !BASE_OPTIMIZER_H */

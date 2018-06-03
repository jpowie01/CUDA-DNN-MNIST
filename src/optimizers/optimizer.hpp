#pragma once
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cstdio>
#include <cmath>

#include "../tensor/tensor2d.cuh"

class Optimizer {
public:
    virtual void optimize(Layer* layer) = 0;
};

#endif  /* !OPTIMIZER_HPP */

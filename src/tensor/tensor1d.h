#pragma once
#ifndef TENSOR1D_H
#define TENSOR1D_H

#include <cstdio>

class Tensor1D {
public:
    // TODO: Make me private!
    int  size;
    float* devData;

    Tensor1D(int size);
    Tensor1D(int size, float* data);
    ~Tensor1D();

    float* getDeviceData();
    float* fetchDataFromDevice();
    
    void add(Tensor1D* tensor);
    void subtract(Tensor1D* tensor);
    void scale(float factor);
};

#endif  /* !TENSOR1D_H */

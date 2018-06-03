#pragma once
#ifndef TENSOR1D_HPP
#define TENSOR1D_HPP

#include <cstdio>

class Tensor1D {
private:
    int size;
    float* devData;

public:
    Tensor1D(int size);
    Tensor1D(int size, float* data);
    ~Tensor1D();

    int getSize();
    float* getDeviceData();
    float* fetchDataFromDevice();
    
    void add(Tensor1D* tensor);
    void subtract(Tensor1D* tensor);
    void scale(float factor);
};

#endif  /* !TENSOR1D_HPP */
